//! G6 allocator micro-benchmark: measure ferro-level allocation overhead with
//! the caching allocator ON vs a pass-through baseline, on a repeated
//! forward "layer" step (matmul -> bias add -> relu) at fixed shapes.
//!
//! Honest-number rules: warm up first (kernel compile + pool fill), then time
//! a large fixed number of steps for each arm, synchronizing via a host
//! read-back each step so wall time reflects real completion. Reports the
//! per-step wall time for both arms and the allocator's freelist hit stats.
//! This is NOT a torch comparison — it isolates ferro's own alloc path.
//!
//! Run: cargo run -p ferro-cuda --release --example alloc_bench

use ferro_core::dispatch::{Backend, BinaryKind, UnaryKind};
use ferro_cuda::CudaBackend;
use std::sync::Arc;
use std::time::Instant;

const M: usize = 256;
const K: usize = 256;
const N: usize = 256;
const WARMUP: usize = 20;
const ITERS: usize = 2000;

fn build_inputs(
    backend: &CudaBackend,
) -> (
    Box<dyn ferro_core::dispatch::DeviceBuffer>,
    Box<dyn ferro_core::dispatch::DeviceBuffer>,
    Box<dyn ferro_core::dispatch::DeviceBuffer>,
) {
    let x = backend
        .alloc_from_host(
            &(0..M * K)
                .map(|i| (i as f32 % 7.0) - 3.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let w = backend
        .alloc_from_host(
            &(0..K * N)
                .map(|i| (i as f32 % 5.0) - 2.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let b = backend
        .alloc_from_host(
            &(0..M * N)
                .map(|i| (i as f32 % 3.0) - 1.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    (x, w, b)
}

fn run_arm(label: &str, backend: Arc<CudaBackend>) {
    let (x, w, b) = build_inputs(&backend);
    let step = || {
        let mm = backend
            .matmul_dev(x.as_ref(), w.as_ref(), M, K, N, false, false)
            .unwrap();
        let bias = backend
            .binary_dev(BinaryKind::Add, mm.as_ref(), b.as_ref())
            .unwrap();
        let act = backend.unary_dev(UnaryKind::Relu, bias.as_ref()).unwrap();
        // Host read-back forces stream completion so timing is real.
        backend.copy_to_host(act.as_ref()).unwrap()
    };

    for _ in 0..WARMUP {
        let _ = step();
    }
    let before = backend.alloc_stats();
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = step();
    }
    let elapsed = t0.elapsed();
    let delta = backend.alloc_stats().since(&before);
    let per_step_us = elapsed.as_secs_f64() * 1e6 / ITERS as f64;
    println!(
        "{label:<14} {per_step_us:8.2} us/step   requests={} hits={} driver_allocs={} pooled={}",
        delta.requests,
        delta.hits,
        delta.driver_allocs,
        backend.pooled_buffers()
    );
}

fn main() {
    if !ferro_cuda::is_available() {
        eprintln!("no CUDA device available; skipping alloc_bench");
        return;
    }
    println!(
        "G6 alloc bench: matmul({M}x{K}x{N}) -> +bias -> relu, {ITERS} steps after {WARMUP} warmup\n"
    );
    match CudaBackend::new(0) {
        Ok(b) => run_arm("caching", Arc::new(b)),
        Err(e) => {
            eprintln!("caching backend init failed: {e}");
            return;
        }
    }
    match CudaBackend::new_passthrough(0) {
        Ok(b) => run_arm("passthrough", Arc::new(b)),
        Err(e) => eprintln!("passthrough backend init failed: {e}"),
    }
    println!(
        "\nNote: cudarc already routes through the driver's cudaMallocAsync pool,\n\
         so 'passthrough' is not naive cudaMalloc/cudaFree - it still hits the\n\
         driver pool. This isolates ferro's own freelist vs the driver call path."
    );
}
