//! Times ferro-core's naive matmul against ferro-fastcpu on square sizes.
//! Run with: cargo run -p ferro-fastcpu --bin bench --release

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

fn main() {
    println!("{:>14} {:>12} {:>12} {:>9}", "size", "naive (ms)", "fast (ms)", "speedup");
    for s in [256usize, 512] {
        let a = lcg_fill(1, s * s);
        let b = lcg_fill(2, s * s);
        let tn = best_of(3, || ferro_core::dispatch::naive_matmul(&a, &b, s, s, s));
        let tf = best_of(5, || ferro_fastcpu::matmul(&a, &b, s, s, s));
        let label = format!("{s}x{s}x{s}");
        println!("{:>14} {:>12.3} {:>12.3} {:>8.2}x", label, tn * 1e3, tf * 1e3, tn / tf);
    }
}
