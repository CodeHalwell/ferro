//! Roofline harness: measures this machine's actual memory-bandwidth and
//! compute-flop ceilings, then reports every op benchmark as a percent of the
//! relevant ceiling (CAPABILITY.md 5.1: attainable = min(peak_flops, AI x
//! bandwidth), so a regression names the resource it lost). Naive kernels
//! (conv2d, bmm, scalar elementwise) are expected to show tiny %-of-roof
//! today; that is the honest baseline this harness exists to record.
//! Run with: cargo run -p ferro-fastcpu --bin roofline --release

use std::hint::black_box;
use std::thread;
use std::time::Instant;

use ferro_core::nn::{cross_entropy_indices, Linear, Module, Relu, Sequential};
use ferro_core::optim::Sgd;
use ferro_core::{Rng, Tensor};

fn lcg_fill(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// Warms up, then times `runs` repetitions and returns the median in
/// milliseconds (median over mean/min: robust to one-off scheduler jitter).
fn median_ms(warmup: usize, runs: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut samples: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2] * 1e3
}

struct Row {
    op: &'static str,
    shape: String,
    ms: f64,
    value: f64,
    unit: &'static str,
    pct: f64,
}

fn gflops_row(op: &'static str, shape: String, ms: f64, flops: f64, roof_gflops: f64) -> Row {
    let gflops = flops / (ms / 1e3) / 1e9;
    Row { op, shape, ms, value: gflops, unit: "GFLOP/s", pct: gflops / roof_gflops * 100.0 }
}

fn gbps_row(op: &'static str, shape: String, ms: f64, bytes: f64, roof_gbps: f64) -> Row {
    let gbps = bytes / (ms / 1e3) / 1e9;
    Row { op, shape, ms, value: gbps, unit: "GB/s", pct: gbps / roof_gbps * 100.0 }
}

// --- roofline measurements -------------------------------------------------

/// Triad c[i] = a[i] + s*b[i] over a working set well above a typical L3 (3
/// arrays x N x 4 bytes ~= 288 MB, >= 256 MB and several times any desktop/
/// server L3), split across all available threads via thread::scope so the
/// number reflects aggregate DRAM bandwidth rather than one core's share.
/// Bytes moved per pass: read a, read b, write c = 3 * N * 4.
///
/// Also measures single-threaded memcpy (read a, write c = 2 * N * 4 bytes)
/// on the same buffers as a reference point between the two.
fn triad_and_memcpy_gbps(threads: usize) -> (f64, f64) {
    let n = 24_000_000usize;
    let a = lcg_fill(31, n);
    let b = lcg_fill(37, n);
    let mut c = vec![0f32; n];
    let s = 1.0001f32;
    let per = n.div_ceil(threads.max(1));

    let triad_ms = median_ms(1, 5, || {
        black_box(&a);
        black_box(&b);
        thread::scope(|scope| {
            for (t, chunk) in c.chunks_mut(per).enumerate() {
                let lo = t * per;
                let a = &a;
                let b = &b;
                scope.spawn(move || {
                    for (i, cv) in chunk.iter_mut().enumerate() {
                        *cv = a[lo + i] + s * b[lo + i];
                    }
                });
            }
        });
        black_box(&c);
    });
    let triad_bytes = 3.0 * n as f64 * 4.0;
    let triad_gbps = triad_bytes / (triad_ms / 1e3) / 1e9;

    let memcpy_ms = median_ms(1, 5, || {
        black_box(&a);
        c.copy_from_slice(&a);
        black_box(&c);
    });
    let memcpy_bytes = 2.0 * n as f64 * 4.0;
    let memcpy_gbps = memcpy_bytes / (memcpy_ms / 1e3) / 1e9;

    (triad_gbps, memcpy_gbps)
}

const FMA_CHAINS: f64 = 8.0;
const FMA_ITERS: u64 = 80_000_000;

/// Unrolled FMA microloop: 8 independent __m256 accumulators (ymm0-ymm7) so
/// the 2 FMA ports issue every cycle instead of stalling on FMA's multi-cycle
/// latency. Each accumulator settles to the fixed point of x*0.999+0.001 (=
/// 1.0), so values never overflow or decay into denormals, keeping every FMA
/// at full speed for the whole run.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_microloop_avx2(iters: u64) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // iters and the seed are compile-time literals at the call site; without
    // black_box here LLVM proves the whole trip count is redundant (the
    // recurrence hits its f32 fixed point in ~30 iterations and then every
    // remaining FMA is a no-op) and deletes the loop entirely.
    let iters = black_box(iters);
    let mul = _mm256_set1_ps(black_box(0.999));
    let add = _mm256_set1_ps(black_box(0.001));
    let seed = _mm256_set1_ps(black_box(1.0));
    let (mut a0, mut a1, mut a2, mut a3) = (seed, seed, seed, seed);
    let (mut a4, mut a5, mut a6, mut a7) = (seed, seed, seed, seed);
    for _ in 0..iters {
        a0 = _mm256_fmadd_ps(a0, mul, add);
        a1 = _mm256_fmadd_ps(a1, mul, add);
        a2 = _mm256_fmadd_ps(a2, mul, add);
        a3 = _mm256_fmadd_ps(a3, mul, add);
        a4 = _mm256_fmadd_ps(a4, mul, add);
        a5 = _mm256_fmadd_ps(a5, mul, add);
        a6 = _mm256_fmadd_ps(a6, mul, add);
        a7 = _mm256_fmadd_ps(a7, mul, add);
    }
    let s01 = _mm256_add_ps(a0, a1);
    let s23 = _mm256_add_ps(a2, a3);
    let s45 = _mm256_add_ps(a4, a5);
    let s67 = _mm256_add_ps(a6, a7);
    _mm256_add_ps(_mm256_add_ps(s01, s23), _mm256_add_ps(s45, s67))
}

/// Portable fallback for non-AVX2 targets: same 8-chain shape via
/// f32::mul_add so the loop still measures FMA-bound (not memory-bound)
/// throughput.
fn fma_microloop_scalar(iters: u64) -> f32 {
    let iters = black_box(iters);
    let (mul, add, seed): (f32, f32, f32) = (black_box(0.999), black_box(0.001), black_box(1.0));
    let (mut a0, mut a1, mut a2, mut a3) = (seed, seed, seed, seed);
    let (mut a4, mut a5, mut a6, mut a7) = (seed, seed, seed, seed);
    for _ in 0..iters {
        a0 = a0.mul_add(mul, add);
        a1 = a1.mul_add(mul, add);
        a2 = a2.mul_add(mul, add);
        a3 = a3.mul_add(mul, add);
        a4 = a4.mul_add(mul, add);
        a5 = a5.mul_add(mul, add);
        a6 = a6.mul_add(mul, add);
        a7 = a7.mul_add(mul, add);
    }
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
}

/// Single-core FMA-bound GFLOP/s: each FMA is 2 flops (multiply, add), times
/// 8 lanes under AVX2, times 8 independent chains, times iteration count.
fn compute_peak_gflops_single_core() -> f64 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        let ms = median_ms(1, 5, || {
            // SAFETY: avx2 and fma were just detected at runtime.
            let r = unsafe { fma_microloop_avx2(FMA_ITERS) };
            black_box(r);
        });
        let flops = FMA_ITERS as f64 * FMA_CHAINS * 8.0 * 2.0;
        return flops / (ms / 1e3) / 1e9;
    }
    let ms = median_ms(1, 5, || {
        black_box(fma_microloop_scalar(FMA_ITERS));
    });
    let flops = FMA_ITERS as f64 * FMA_CHAINS * 2.0;
    flops / (ms / 1e3) / 1e9
}

// --- op benchmarks ----------------------------------------------------------

/// Matmul FLOPs: one multiply and one add per output element per k step,
/// i.e. 2*M*K*N. Sizes chosen so the largest (2048^3, ~17.2 GFLOP) still
/// finishes comfortably under the ~30s per-measurement budget.
fn bench_matmul(rows: &mut Vec<Row>, compute_roof: f64) {
    for &size in &[512usize, 1024, 2048] {
        let a = Tensor::from_vec(lcg_fill(size as u64 * 3 + 1, size * size), &[size, size]).unwrap();
        let b = Tensor::from_vec(lcg_fill(size as u64 * 7 + 2, size * size), &[size, size]).unwrap();
        let ms = median_ms(1, 5, || {
            black_box(&a);
            black_box(&b);
            let out = a.matmul(&b).unwrap();
            black_box(out);
        });
        let flops = 2.0 * (size as f64).powi(3);
        rows.push(gflops_row("matmul", format!("{size}x{size}x{size}"), ms, flops, compute_roof));
    }
}

/// 32M-element f32 tensors. Byte accounting per CAPABILITY.md 5.1: a binary
/// op reads two operands and writes one output (2 reads + 1 write = 12
/// bytes/elem); a unary op reads one and writes one (1 read + 1 write = 8
/// bytes/elem). Today's CpuBackend unary/binary kernels (dispatch.rs) are
/// plain scalar iterator maps with no threading, so low %-of-bandwidth here
/// is the expected honest baseline, not a bug.
fn bench_elementwise(rows: &mut Vec<Row>, bw_roof: f64) {
    let n = 32_000_000usize;
    let shape = format!("{n} f32");
    let a = Tensor::from_vec(lcg_fill(41, n), &[n]).unwrap();
    let b = Tensor::from_vec(lcg_fill(43, n), &[n]).unwrap();

    let ms = median_ms(1, 5, || {
        black_box(&a);
        black_box(&b);
        let out = a.add(&b).unwrap();
        black_box(out);
    });
    rows.push(gbps_row("add", shape.clone(), ms, n as f64 * 12.0, bw_roof));

    let ms = median_ms(1, 5, || {
        black_box(&a);
        black_box(&b);
        let out = a.mul(&b).unwrap();
        black_box(out);
    });
    rows.push(gbps_row("mul", shape.clone(), ms, n as f64 * 12.0, bw_roof));

    let ms = median_ms(1, 5, || {
        black_box(&a);
        let out = a.relu();
        black_box(out);
    });
    rows.push(gbps_row("relu", shape.clone(), ms, n as f64 * 8.0, bw_roof));

    let ms = median_ms(1, 5, || {
        black_box(&a);
        let out = a.exp();
        black_box(out);
    });
    rows.push(gbps_row("exp", shape, ms, n as f64 * 8.0, bw_roof));
}

/// sum() over 32M elements logically reads the buffer once (n*4 bytes). The
/// current host path (tensor.rs `to_vec` -> `gather`) copies the contiguous
/// buffer before reducing it, so the real op touches more bytes than this
/// credits it for; the achieved GB/s below is therefore a conservative (not
/// inflated) read of how far off a fused single-pass reduction still is.
fn bench_sum(rows: &mut Vec<Row>, bw_roof: f64) {
    let n = 32_000_000usize;
    let a = Tensor::from_vec(lcg_fill(53, n), &[n]).unwrap();
    let ms = median_ms(1, 5, || {
        black_box(&a);
        let out = a.sum();
        black_box(out);
    });
    rows.push(gbps_row("sum", format!("{n} f32"), ms, n as f64 * 4.0, bw_roof));
}

/// bmm FLOPs: 2*batch*M*K*N (multiply-add per output element per k step).
fn bench_bmm(rows: &mut Vec<Row>, compute_roof: f64) {
    let (batch, m, k, n) = (16usize, 128usize, 128usize, 128usize);
    let a = Tensor::from_vec(lcg_fill(61, batch * m * k), &[batch, m, k]).unwrap();
    let b = Tensor::from_vec(lcg_fill(67, batch * k * n), &[batch, k, n]).unwrap();
    let ms = median_ms(1, 5, || {
        black_box(&a);
        black_box(&b);
        let out = a.bmm(&b).unwrap();
        black_box(out);
    });
    let flops = 2.0 * batch as f64 * m as f64 * k as f64 * n as f64;
    rows.push(gflops_row("bmm", format!("{batch}x{m}x{k}x{n}"), ms, flops, compute_roof));
}

/// conv2d FLOPs: 2*N*Cout*Cin*OH*OW*KH*KW (one multiply-add per output tap
/// per input channel per kernel position).
fn bench_conv2d(rows: &mut Vec<Row>, compute_roof: f64) {
    let (n, cin, cout, h, w, kh, kw, stride, pad) = (4usize, 32usize, 64usize, 56usize, 56usize, 3usize, 3usize, 1usize, 1usize);
    let x = Tensor::from_vec(lcg_fill(71, n * cin * h * w), &[n, cin, h, w]).unwrap();
    let wt = Tensor::from_vec(lcg_fill(73, cout * cin * kh * kw), &[cout, cin, kh, kw]).unwrap();
    let ms = median_ms(1, 5, || {
        black_box(&x);
        black_box(&wt);
        let out = x.conv2d(&wt, stride, pad).unwrap();
        black_box(out);
    });
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (w + 2 * pad - kw) / stride + 1;
    let flops = 2.0 * n as f64 * cout as f64 * cin as f64 * oh as f64 * ow as f64 * kh as f64 * kw as f64;
    let shape = format!("n{n} cin{cin} cout{cout} {h}x{w} k{kh}x{kw} s{stride} p{pad}");
    rows.push(gflops_row("conv2d", shape, ms, flops, compute_roof));
}

/// One MLP train step (784-256-10, batch 256): forward through two Linear
/// layers + relu + cross_entropy, backward through the same graph, then an
/// Sgd step. FLOPs counted are the two Linear matmuls: a forward matmul plus
/// its dA/dB backward matmuls each cost the same as the forward one, so 3x
/// forward matmul flops; relu/softmax/bias/SGD are O(batch*width), negligible
/// next to the O(batch*width*width) matmuls.
fn bench_mlp_step(rows: &mut Vec<Row>, compute_roof: f64) {
    let (batch, d0, d1, d2) = (256usize, 784usize, 256usize, 10usize);
    let rng = Rng::new(99);
    let x = Tensor::from_vec(lcg_fill(81, batch * d0), &[batch, d0]).unwrap();
    let targets = Tensor::from_vec_i64((0..batch as i64).map(|i| i % d2 as i64).collect(), &[batch]).unwrap();
    let model = Sequential::new(vec![
        Box::new(Linear::new(d0, d1, &rng)),
        Box::new(Relu),
        Box::new(Linear::new(d1, d2, &rng)),
    ]);
    let mut opt = Sgd::new(model.parameters(), 0.01);

    let ms = median_ms(1, 5, || {
        opt.zero_grad();
        let logits = model.forward(&x).unwrap();
        let loss = cross_entropy_indices(&logits, &targets).unwrap();
        loss.backward();
        opt.step();
        black_box(&loss);
    });
    let matmul_flops = 2.0 * batch as f64 * d0 as f64 * d1 as f64 + 2.0 * batch as f64 * d1 as f64 * d2 as f64;
    let step_flops = matmul_flops * 3.0;
    let steps_per_sec = 1e3 / ms;
    let shape = format!("{d0}-{d1}-{d2} batch{batch} ({steps_per_sec:.1} steps/s)");
    rows.push(gflops_row("mlp_train_step", shape, ms, step_flops, compute_roof));
}

fn main() {
    let cores = thread::available_parallelism().map_or(1, |p| p.get());
    let (bw_roof, memcpy_roof) = triad_and_memcpy_gbps(cores);
    let compute_single = compute_peak_gflops_single_core();
    let compute_roof = compute_single * cores as f64;

    println!(
        "measured roofline: bandwidth {bw_roof:.2} GB/s ({cores} threads, triad), memcpy {memcpy_roof:.2} GB/s (1 thread) \
         | compute {compute_single:.2} GFLOP/s/core x {cores} cores = {compute_roof:.2} GFLOP/s peak"
    );
    println!();

    ferro_fastcpu::install();

    let mut rows = Vec::new();
    bench_matmul(&mut rows, compute_roof);
    bench_elementwise(&mut rows, bw_roof);
    bench_sum(&mut rows, bw_roof);
    bench_bmm(&mut rows, compute_roof);
    bench_conv2d(&mut rows, compute_roof);
    bench_mlp_step(&mut rows, compute_roof);

    println!("| op | shape | median ms | achieved | % of roof |");
    println!("|---|---|---|---|---|");
    for r in &rows {
        println!("| {} | {} | {:.3} | {:.2} {} | {:.1}% |", r.op, r.shape, r.ms, r.value, r.unit, r.pct);
    }
}
