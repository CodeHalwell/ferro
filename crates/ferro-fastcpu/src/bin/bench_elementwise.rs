//! GB/s for elementwise kernels at 1M and 32M f32: CpuBackend baseline vs
//! FastCpuBackend single- and multi-threaded, plus a memcpy bandwidth
//! measurement so the numbers read as %-of-bandwidth (docs/CAPABILITY.md
//! 5.1: elementwise ops are bandwidth-bound, AI ~ 1/12 flop/byte, so no
//! elementwise kernel is ever compute-bound - the only lever is bytes moved).
//! Run with: cargo run -p ferro-fastcpu --release --bin bench_elementwise

use std::time::Instant;

use ferro_core::dispatch::{Backend, BinaryKind, UnaryKind};
use ferro_core::CpuBackend;
use ferro_fastcpu::elementwise::{self, FastCpuBackend};

fn lcg_fill(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn best_of(runs: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn gbs(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

fn memcpy_bandwidth(n: usize) -> f64 {
    let src = lcg_fill(1, n);
    let mut dst = vec![0f32; n];
    let secs = best_of(7, || {
        dst.copy_from_slice(std::hint::black_box(&src));
    });
    std::hint::black_box(&dst);
    gbs(n * 4 * 2, secs)
}

fn report(name: &str, n: usize, bytes: usize, secs: f64, roof: f64) {
    let bw = gbs(bytes, secs);
    println!(
        "  {:<10} n={:<10} {:>8.2} GB/s  {:>5.1}% of roof  ({:>7.2} ms)",
        name,
        n,
        bw,
        100.0 * bw / roof,
        secs * 1e3
    );
}

fn bench_binary(name: &str, kind: BinaryKind, a: &[f32], b: &[f32], n: usize, roof: f64) {
    let bytes = n * 4 * 3;
    let t_cpu = best_of(5, || {
        std::hint::black_box(CpuBackend.binary(kind, a, b));
    });
    let t_serial = best_of(5, || {
        std::hint::black_box(elementwise::binary_serial(kind, a, b));
    });
    let t_fast = best_of(5, || {
        std::hint::black_box(FastCpuBackend.binary(kind, a, b));
    });
    report(&format!("{name}/cpu"), n, bytes, t_cpu, roof);
    report(&format!("{name}/fast-1t"), n, bytes, t_serial, roof);
    report(&format!("{name}/fast-Nt"), n, bytes, t_fast, roof);
    println!("    speedup vs CpuBackend: {:.2}x (1t), {:.2}x (Nt)", t_cpu / t_serial, t_cpu / t_fast);
}

fn bench_unary(name: &str, kind: UnaryKind, x: &[f32], n: usize, roof: f64) {
    let bytes = n * 4 * 2;
    let t_cpu = best_of(5, || {
        std::hint::black_box(CpuBackend.unary(kind, x));
    });
    let t_serial = best_of(5, || {
        std::hint::black_box(elementwise::unary_serial(kind, x));
    });
    let t_fast = best_of(5, || {
        std::hint::black_box(FastCpuBackend.unary(kind, x));
    });
    report(&format!("{name}/cpu"), n, bytes, t_cpu, roof);
    report(&format!("{name}/fast-1t"), n, bytes, t_serial, roof);
    report(&format!("{name}/fast-Nt"), n, bytes, t_fast, roof);
    println!("    speedup vs CpuBackend: {:.2}x (1t), {:.2}x (Nt)", t_cpu / t_serial, t_cpu / t_fast);
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |p| p.get());
    println!("available_parallelism = {threads}\n");

    for &n in &[1usize << 20, 1 << 25] {
        let roof = memcpy_bandwidth(n);
        println!("=== n = {n} ({:.1} MiB/array), memcpy roof = {:.2} GB/s ===", (n * 4) as f64 / (1 << 20) as f64, roof);
        let a = lcg_fill(10, n);
        let b = lcg_fill(20, n);
        bench_binary("add", BinaryKind::Add, &a, &b, n, roof);
        bench_binary("mul", BinaryKind::Mul, &a, &b, n, roof);
        bench_unary("relu", UnaryKind::Relu, &a, n, roof);
        bench_unary("exp", UnaryKind::Exp, &a, n, roof);
        println!();
    }
}
