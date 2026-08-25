//! Micro-benchmark: fused pointwise chain (one launch) vs the unfused
//! equivalent (three launches with global-memory intermediates). A device
//! sync after each timed loop makes timings measure completed GPU work.

use std::time::Instant;

use ferro_core::{Backend, BinaryKind, UnaryKind};
use ferro_cuda::{chain_source, ChainStep};

fn main() {
    if !ferro_cuda::is_available() {
        eprintln!("no CUDA device");
        return;
    }
    let b = std::sync::Arc::new(ferro_cuda::CudaBackend::new(0).unwrap());
    let n: usize = std::env::args()
        .skip_while(|a| a != "--n")
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1usize << 22);
    let x: Vec<f32> = (0..n).map(|i| ((i % 31) as f32 - 15.0) * 0.2).collect();
    let y: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.3).collect();
    let z: Vec<f32> = (0..n).map(|i| i as f32 * 1e-6 - 1.0).collect();

    let steps = vec![
        ChainStep::Unary(UnaryKind::Gelu),
        ChainStep::Binary {
            kind: BinaryKind::Mul,
            other: 1,
        },
        ChainStep::Binary {
            kind: BinaryKind::Add,
            other: 2,
        },
    ];
    let _src = chain_source(&steps);

    const ITERS: u32 = 200;
    let xd = b.alloc_from_host(&x).unwrap();
    let yd = b.alloc_from_host(&y).unwrap();
    let zd = b.alloc_from_host(&z).unwrap();

    // Capture FIRST, before any other stream work, to isolate whether prior
    // launches break end_capture.
    let captured_early = match b.capture_chain(&steps, &[xd.as_ref(), yd.as_ref(), zd.as_ref()]) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("early capture failed: {e}");
            None
        }
    };

    // Correctness anchor: fused result must match the unfused reference.
    let fused_ref = b.chain_res(&steps, &[&x, &y, &z]).unwrap();
    let g0 = b.unary_dev(UnaryKind::Gelu, xd.as_ref()).unwrap();
    let m0 = b
        .binary_dev(BinaryKind::Mul, g0.as_ref(), yd.as_ref())
        .unwrap();
    let a0 = b
        .binary_dev(BinaryKind::Add, m0.as_ref(), zd.as_ref())
        .unwrap();
    let unfused_ref = b.copy_to_host(a0.as_ref()).unwrap();
    for i in [0, n / 2, n - 1] {
        assert!(
            (fused_ref[i] - unfused_ref[i]).abs() <= 1e-5 * fused_ref[i].abs().max(1.0),
            "elem {i}: fused {} unfused {}",
            fused_ref[i],
            unfused_ref[i]
        );
    }

    let sync = |b: &ferro_cuda::CudaBackend| {
        let probe = b.alloc_from_host(&[0.0f32]).unwrap();
        let _ = b.copy_to_host(probe.as_ref()).unwrap();
    };
    let mut warm = None;
    for _ in 0..10 {
        warm = Some(
            b.chain_dev(&steps, &[xd.as_ref(), yd.as_ref(), zd.as_ref()])
                .unwrap(),
        );
    }
    drop(warm);
    sync(&b);

    let t = Instant::now();
    for _ in 0..ITERS {
        let out = b
            .chain_dev(&steps, &[xd.as_ref(), yd.as_ref(), zd.as_ref()])
            .unwrap();
        std::hint::black_box(out);
    }
    sync(&b);
    let fused = t.elapsed();

    let t = Instant::now();
    for _ in 0..ITERS {
        let g = b.unary_dev(UnaryKind::Gelu, xd.as_ref()).unwrap();
        let m = b
            .binary_dev(BinaryKind::Mul, g.as_ref(), yd.as_ref())
            .unwrap();
        let a = b
            .binary_dev(BinaryKind::Add, m.as_ref(), zd.as_ref())
            .unwrap();
        std::hint::black_box(a);
    }
    sync(&b);
    let unfused = t.elapsed();

    println!("n = {n} ({:.1} MiB/intermediate), iters = {ITERS}", n as f64 * 4.0 / 1048576.0);
    println!(
        "fused   (1 launch): {:>9.2?} total, {:>8.3?}/iter",
        fused,
        fused / ITERS
    );
    println!(
        "unfused (3 launches): {:>7.2?} total, {:>8.3?}/iter",
        unfused,
        unfused / ITERS
    );
    println!(
        "speedup: {:.2}x",
        unfused.as_secs_f64() / fused.as_secs_f64()
    );

    // CUDA-graph replay of the same fused chain: capture once, then each
    // iteration is a single graph launch writing to the same output buffer.
    let captured = match b.capture_chain(&steps, &[xd.as_ref(), yd.as_ref(), zd.as_ref()]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("capture error detail: {e}");
            return;
        }
    };
    // Correctness anchor: one replay matches the eager fused result. The
    // sync first - replay is async on the stream.
    sync(&b);
    captured.replay().unwrap();
    sync(&b);
    let got = captured.copy_output_to_host(&b).unwrap();
    for i in [0, n / 2, n - 1] {
        assert!(
            (got[i] - fused_ref[i]).abs() <= 1e-5 * fused_ref[i].abs().max(1.0),
            "elem {i}: replay {} eager {}",
            got[i],
            fused_ref[i]
        );
    }

    let t = Instant::now();
    for _ in 0..ITERS {
        captured.replay().unwrap();
        std::hint::black_box(());
    }
    sync(&b);
    let graphed = t.elapsed();

    println!(
        "graph   (1 launch): {:>9.2?} total, {:>8.3?}/iter",
        graphed,
        graphed / ITERS
    );
    println!(
        "graph vs fused: {:.2}x, graph vs unfused: {:.2}x",
        fused.as_secs_f64() / graphed.as_secs_f64(),
        unfused.as_secs_f64() / graphed.as_secs_f64()
    );
}
