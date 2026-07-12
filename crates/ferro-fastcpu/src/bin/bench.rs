//! Times ferro-core's naive matmul against ferro-fastcpu, reporting GFLOP/s
//! and %-of-measured-peak for square and skinny shapes, single- and
//! multi-threaded. Run with: cargo run -p ferro-fastcpu --bin bench --release

use std::time::Instant;

fn lcg_fill(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn best_of(runs: usize, mut f: impl FnMut() -> Vec<f32>) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        std::hint::black_box(f());
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// Pure-FMA register microloop: 8 independent ymm accumulator chains over a
/// fixed instruction count, enough independent work to saturate both FMA
/// ports and hide their latency. Used as a measured roofline (rather than
/// trusting nameplate clock x lanes x ports, which turbo/throttling and
/// virtualization make unreliable).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_roofline_avx2(iters: u64) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = [_mm256_set1_ps(1.0000001); 8];
    let mul = _mm256_set1_ps(1.0000002);
    for _ in 0..iters {
        for a in acc.iter_mut() {
            *a = _mm256_fmadd_ps(*a, mul, mul);
        }
    }
    let mut sum = 0f32;
    for a in acc {
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), a);
        sum += lanes.iter().sum::<f32>();
    }
    sum
}

/// Measured single-core FMA roofline in GFLOP/s. 8 chains x one 8-wide FMA
/// (2 flops/lane) per iteration = 128 flops/iter.
fn measured_fma_peak_gflops_per_core() -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let iters = 300_000_000u64;
        let flops = iters as f64 * 128.0;
        let t = Instant::now();
        std::hint::black_box(unsafe { fma_roofline_avx2(iters) });
        let secs = t.elapsed().as_secs_f64();
        return flops / secs / 1e9;
    }
    f64::NAN
}

struct Shape {
    label: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

fn main() {
    let peak1 = measured_fma_peak_gflops_per_core();
    let cores = std::thread::available_parallelism().map_or(1, |p| p.get());
    println!("measured FMA roofline: {peak1:.1} GFLOP/s/core x {cores} cores = {:.1} GFLOP/s", peak1 * cores as f64);
    println!();

    let shapes = [
        Shape { label: "256x256x256", m: 256, k: 256, n: 256 },
        Shape { label: "512x512x512", m: 512, k: 512, n: 512 },
        Shape { label: "1024x1024x1024", m: 1024, k: 1024, n: 1024 },
        Shape { label: "2048x2048x2048", m: 2048, k: 2048, n: 2048 },
        Shape { label: "64x4096x64", m: 64, k: 4096, n: 64 },
        Shape { label: "2048x64x2048", m: 2048, k: 64, n: 2048 },
    ];

    println!(
        "{:>16} {:>4} {:>12} {:>12} {:>9} {:>10} {:>7}",
        "shape", "thr", "naive (ms)", "fast (ms)", "speedup", "GFLOP/s", "%peak"
    );
    for shape in &shapes {
        let Shape { label, m, k, n } = *shape;
        let a = lcg_fill(1, m * k);
        let b = lcg_fill(2, k * n);
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        let naive_runs = if m * k * n > 1 << 24 { 1 } else { 3 };
        let tn = best_of(naive_runs, || ferro_core::dispatch::naive_matmul(&a, &b, m, k, n));

        for threads in [1usize, cores] {
            let tf = best_of(5, || ferro_fastcpu::matmul_with_threads(&a, &b, m, k, n, threads));
            let gflops = flops / tf / 1e9;
            let pct = gflops / (peak1 * threads as f64) * 100.0;
            let row_label = if threads == 1 { label } else { "" };
            println!(
                "{row_label:>16} {threads:>4} {:>12.3} {:>12.3} {:>8.2}x {gflops:>10.2} {pct:>6.1}%",
                tn * 1e3,
                tf * 1e3,
                tn / tf,
            );
        }
    }
}
