//! Vectorized/threaded elementwise, reduction and softmax CPU backend,
//! registered through `ferro_core::dispatch::register_backend` (see
//! `install_backend` in lib.rs).
//!
//! Numerics against `CpuBackend`:
//! - arithmetic kinds (neg/relu/abs/sqrt/clamp/gtz/powf, every binary op) use
//!   the reference formulas verbatim and are bit-for-bit identical; LLVM
//!   autovectorizes them over chunks_exact(8) under AVX2+FMA
//! - sum and sum_dim replay core's fixed-shape pairwise tree, so they are
//!   bit-for-bit identical too, at any thread count
//! - exp/log/tanh/sigmoid/silu/gelu/gelu_erf/gelu_erf_grad and softmax/
//!   log_softmax use avx2.rs's polynomials: never more than 4 ULP further
//!   from the f64-evaluated formula than CpuBackend's own result
//! - without AVX2+FMA (or off x86_64) everything is the reference formula
//!
//! Threading goes through the persistent pool in workers.rs, never a
//! per-call spawn. Streaming kinds keep PAR_THRESHOLD, measured on the
//! 4-core/33MB-L3 dev machine: below ~1<<21 elements the working set is
//! L3-resident and one core already saturates it (threading was 0.7-0.85x at
//! 1<<20 and >1.4x from 1<<21 on, see bench_elementwise). Transcendentals
//! cost ~10x more per element: threading lost at 32K and won from 64K.

use ferro_core::dispatch::{Backend, BinaryKind, UnaryKind};
use ferro_core::CpuBackend;

use crate::workers;

const PAR_THRESHOLD: usize = 1 << 21;
const PAR_THRESHOLD_MATH: usize = 1 << 16;

pub struct FastCpuBackend;

impl Backend for FastCpuBackend {
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        // Pool-backed; the chunk loops write every element.
        let mut out = ferro_core::pool::take_uninit(x.len());
        let cheap = matches!(kind, UnaryKind::Neg | UnaryKind::Relu | UnaryKind::Sqrt | UnaryKind::Abs | UnaryKind::Clamp { .. } | UnaryKind::Gtz);
        let par = x.len() >= if cheap { PAR_THRESHOLD } else { PAR_THRESHOLD_MATH };
        par_split(&mut out, 1, par, |i, o| unary_chunk(kind, &x[i..i + o.len()], o));
        out
    }

    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
        // zip would silently truncate on a length mismatch; fail loudly instead.
        assert_eq!(a.len(), b.len(), "binary operands must have the same length");
        let mut out = ferro_core::pool::take_uninit(a.len());
        par_split(&mut out, 1, a.len() >= PAR_THRESHOLD, |i, o| binary_chunk(kind, &a[i..i + o.len()], &b[i..i + o.len()], o));
        out
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        crate::matmul(a, b, m, k, n)
    }

    fn matmul_batch(&self, a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        crate::matmul_batch(a, b, batch, m, k, n)
    }

    fn sum(&self, x: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            return crate::avx2::sum(x);
        }
        CpuBackend.sum(x)
    }

    fn sum_dim(&self, x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            return crate::avx2::sum_dim(x, shape, dim);
        }
        CpuBackend.sum_dim(x, shape, dim)
    }

    fn softmax(&self, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            return crate::avx2::softmax(x, rows, cols, false);
        }
        CpuBackend.softmax(x, rows, cols)
    }

    fn log_softmax(&self, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        #[cfg(target_arch = "x86_64")]
        if avx2() {
            return crate::avx2::softmax(x, rows, cols, true);
        }
        CpuBackend.log_softmax(x, rows, cols)
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

struct SyncPtr(*mut f32);
// SAFETY: par_split hands each task a disjoint range of the pointee.
unsafe impl Sync for SyncPtr {}

/// Runs `f(start, chunk)` over `out` cut into contiguous chunks, one per pool
/// thread when `parallel` (else inline as one chunk); every chunk but the last
/// is a whole number of `unit`s.
pub(crate) fn par_split(out: &mut [f32], unit: usize, parallel: bool, f: impl Fn(usize, &mut [f32]) + Sync) {
    let units = out.len().div_ceil(unit.max(1));
    let threads = if parallel { workers::threads().min(units) } else { 1 };
    if threads <= 1 {
        f(0, out);
        return;
    }
    let (len, per) = (out.len(), units.div_ceil(threads) * unit);
    let base = SyncPtr(out.as_mut_ptr());
    let base = &base;
    workers::run(len.div_ceil(per), &|t| {
        let start = t * per;
        // SAFETY: tasks get disjoint in-bounds ranges of `out`, which
        // outlives run().
        let chunk = unsafe { std::slice::from_raw_parts_mut(base.0.add(start), per.min(len - start)) };
        f(start, chunk);
    });
}

/// Vectorized but forced single-threaded; exposed so bench_elementwise can
/// isolate the vectorization win from the threading win.
pub fn unary_serial(kind: UnaryKind, x: &[f32]) -> Vec<f32> {
    let mut out = ferro_core::pool::take_uninit(x.len());
    unary_chunk(kind, x, &mut out);
    out
}

pub fn binary_serial(kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = ferro_core::pool::take_uninit(a.len());
    binary_chunk(kind, a, b, &mut out);
    out
}

fn unary_chunk(kind: UnaryKind, x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: avx2 and fma were just detected at runtime.
        unsafe {
            if !crate::avx2::unary(kind, x, out) {
                unary_chunk_avx2(kind, x, out);
            }
        }
        return;
    }
    unary_chunk_body(kind, x, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn unary_chunk_avx2(kind: UnaryKind, x: &[f32], out: &mut [f32]) {
    unary_chunk_body(kind, x, out);
}

fn binary_chunk(kind: BinaryKind, a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if avx2() {
        // SAFETY: avx2 and fma were just detected at runtime.
        unsafe { binary_chunk_avx2(kind, a, b, out) };
        return;
    }
    binary_chunk_body(kind, a, b, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn binary_chunk_avx2(kind: BinaryKind, a: &[f32], b: &[f32], out: &mut [f32]) {
    binary_chunk_body(kind, a, b, out);
}

#[inline(always)]
fn unary_chunk_body(kind: UnaryKind, x: &[f32], out: &mut [f32]) {
    match kind {
        UnaryKind::Neg => apply1(x, out, |v| -v),
        // Not v.max(0.0): f32::max drops NaN, torch's relu propagates it.
        UnaryKind::Relu => apply1(x, out, |v| if v > 0.0 || v.is_nan() { v } else { 0.0 }),
        UnaryKind::Exp => apply1(x, out, |v| v.exp()),
        UnaryKind::Sigmoid => apply1(x, out, |v| 1.0 / (1.0 + (-v).exp())),
        UnaryKind::Tanh => apply1(x, out, |v| v.tanh()),
        UnaryKind::Sqrt => apply1(x, out, |v| v.sqrt()),
        UnaryKind::Abs => apply1(x, out, |v| v.abs()),
        UnaryKind::Log => apply1(x, out, |v| v.ln()),
        UnaryKind::Powf(p) => apply1(x, out, |v| v.powf(p)),
        // max/min chain, not f32::clamp (which panics on min > max); matches
        // torch: min > max yields max everywhere. NaN passes through.
        UnaryKind::Clamp { min, max } => apply1(x, out, |v| if v.is_nan() { v } else { v.max(min).min(max) }),
        UnaryKind::Gtz => apply1(x, out, |v| if v > 0.0 { 1.0 } else { 0.0 }),
        // Same tanh approximation as the core reference implementation.
        UnaryKind::Gelu => {
            apply1(x, out, |v| {
                let u = 0.797_884_6 * (v + 0.044715 * v * v * v);
                0.5 * v * (1.0 + u.tanh())
            })
        }
        // Exact-erf GELU and its derivative: the erf itself comes from the
        // shared core helper, so results are bit-identical to CpuBackend.
        UnaryKind::GeluErf => apply1(x, out, |v| {
            0.5 * v * (1.0 + ferro_core::dispatch::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2))
        }),
        UnaryKind::GeluErfGrad => apply1(x, out, |v| {
            0.5 * (1.0 + ferro_core::dispatch::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2))
                + v * (-0.5 * v * v).exp() * 0.398_942_28
        }),
        UnaryKind::Silu => apply1(x, out, |v| v / (1.0 + (-v).exp())),
    }
}

#[inline(always)]
fn binary_chunk_body(kind: BinaryKind, a: &[f32], b: &[f32], out: &mut [f32]) {
    match kind {
        BinaryKind::Add => apply2(a, b, out, |x, y| x + y),
        BinaryKind::Sub => apply2(a, b, out, |x, y| x - y),
        BinaryKind::Mul => apply2(a, b, out, |x, y| x * y),
        BinaryKind::Div => apply2(a, b, out, |x, y| x / y),
    }
}

/// chunks_exact(8) plus its remainder: the fixed-size array keeps a chunk's
/// results in registers so LLVM can vectorize the arithmetic-only formulas;
/// transcendental formulas fall back to per-lane scalar calls within the
/// same loop shape.
#[inline(always)]
fn apply1(x: &[f32], out: &mut [f32], f: impl Fn(f32) -> f32) {
    // zip would silently truncate on a length mismatch; fail loudly instead.
    assert_eq!(x.len(), out.len(), "unary output length must match input");
    let xchunks = x.chunks_exact(8);
    let rem = xchunks.remainder();
    let main = x.len() - rem.len();
    for (xc, oc) in x[..main].chunks_exact(8).zip(out[..main].chunks_exact_mut(8)) {
        let mut r = [0f32; 8];
        for j in 0..8 {
            r[j] = f(xc[j]);
        }
        oc.copy_from_slice(&r);
    }
    for (&xv, ov) in rem.iter().zip(out[main..].iter_mut()) {
        *ov = f(xv);
    }
}

#[inline(always)]
fn apply2(a: &[f32], b: &[f32], out: &mut [f32], f: impl Fn(f32, f32) -> f32) {
    // zip would silently truncate on a length mismatch; fail loudly instead.
    assert_eq!(a.len(), b.len(), "binary operands must have the same length");
    assert_eq!(a.len(), out.len(), "binary output length must match operands");
    let achunks = a.chunks_exact(8);
    let rem = achunks.remainder();
    let main = a.len() - rem.len();
    let chunks = a[..main].chunks_exact(8).zip(b[..main].chunks_exact(8)).zip(out[..main].chunks_exact_mut(8));
    for ((ac, bc), oc) in chunks {
        let mut r = [0f32; 8];
        for j in 0..8 {
            r[j] = f(ac[j], bc[j]);
        }
        oc.copy_from_slice(&r);
    }
    for ((&av, &bv), ov) in rem.iter().zip(b[main..].iter()).zip(out[main..].iter_mut()) {
        *ov = f(av, bv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod diagnostics { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/bmm_diagnostics.rs")); }
    use ferro_core::{CpuBackend, Tensor};

    fn lcg_fill(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 4.0
            })
            .collect()
    }

    /// Random values with a run of special cases (both signs of zero, NaN,
    /// both infinities, both signs of a denormal) planted at the front and,
    /// length permitting, mirrored at the back so both the main vectorized
    /// loop and the chunks_exact remainder see them.
    fn special_vals(seed: u64, len: usize) -> Vec<f32> {
        let specials = [
            0.0f32,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE / 2.0,
            -f32::MIN_POSITIVE / 2.0,
            1.0,
            -1.0,
            2.5,
        ];
        let head = specials.len().min(len);
        let tail = if len >= 2 * specials.len() { specials.len() } else { 0 };
        let mid = len - head - tail;
        let mut out = Vec::with_capacity(len);
        out.extend_from_slice(&specials[..head]);
        out.extend(lcg_fill(seed, mid));
        out.extend_from_slice(&specials[..tail]);
        out
    }

    fn assert_bitwise(got: &[f32], want: &[f32], ctx: &str) {
        assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
        for (i, (&x, &y)) in got.iter().zip(want).enumerate() {
            // Both NaN counts as parity: the NaN payload bits are not
            // semantically meaningful and can differ between the scalar and
            // vectorized paths without either being wrong.
            if x.is_nan() && y.is_nan() {
                continue;
            }
            assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: mismatch at index {i}: {x} vs {y}");
        }
    }

    // The last two exercise the single-threaded-but-vectorized path (well
    // under PAR_THRESHOLD) and the multithreaded path (just over it), both
    // with a non-8-aligned remainder.
    const LENGTHS: [usize; 11] = [0, 1, 7, 8, 9, 15, 16, 17, 1023, (1 << 20) + 3, PAR_THRESHOLD + 3];

    #[test]
    fn unary_parity_exact_kinds() {
        let kinds = [
            UnaryKind::Neg,
            UnaryKind::Relu,
            UnaryKind::Sqrt,
            UnaryKind::Abs,
            UnaryKind::Powf(0.5),
            UnaryKind::Powf(3.0),
            UnaryKind::Powf(-2.0),
            UnaryKind::Clamp { min: -1.0, max: 1.0 },
            // min > max: torch semantics are max everywhere, no panic.
            UnaryKind::Clamp { min: 2.0, max: 1.0 },
            UnaryKind::Gtz,
        ];
        for (ki, &kind) in kinds.iter().enumerate() {
            for &len in &LENGTHS {
                let x = special_vals(len as u64 * 7 + ki as u64, len);
                let want = CpuBackend.unary(kind, &x);
                let ctx = format!("{kind:?} len={len}");
                assert_bitwise(&FastCpuBackend.unary(kind, &x), &want, &ctx);
                assert_bitwise(&unary_serial(kind, &x), &want, &format!("{ctx} (serial)"));
            }
        }
    }

    const APPROX: [UnaryKind; 8] = [
        UnaryKind::Exp,
        UnaryKind::Log,
        UnaryKind::Tanh,
        UnaryKind::Sigmoid,
        UnaryKind::Silu,
        UnaryKind::Gelu,
        UnaryKind::GeluErf,
        UnaryKind::GeluErfGrad,
    ];

    /// Phi(x) from the reference's A-S 7.1.26 erf, evaluated in f64 without
    /// forming 1 + erf (which cancels in the left tail even in f64).
    fn cdf64(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.3275911 * x.abs() * std::f64::consts::FRAC_1_SQRT_2);
        let p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
        let hq = 0.5 * p * (-0.5 * x * x).exp();
        if x < 0.0 { hq } else { 1.0 - hq }
    }

    /// The kind's formula in f64 with the reference's f32 constants, in
    /// algebraically equal forms that stay accurate in the tails (tanh-GELU
    /// as x * sigmoid(2u)).
    fn truth(kind: UnaryKind, v: f32) -> f64 {
        let x = v as f64;
        match kind {
            UnaryKind::Exp => x.exp(),
            UnaryKind::Log => x.ln(),
            UnaryKind::Tanh => x.tanh(),
            UnaryKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            UnaryKind::Silu => x / (1.0 + (-x).exp()),
            UnaryKind::Gelu => x / (1.0 + (-2.0 * 0.797_884_6f32 as f64 * (x + 0.044715f32 as f64 * x * x * x)).exp()),
            UnaryKind::GeluErf => x * cdf64(x),
            UnaryKind::GeluErfGrad => cdf64(x) + x * (-0.5 * x * x).exp() * 0.398_942_28f32 as f64,
            _ => unreachable!(),
        }
    }

    /// Size of one f32 ULP at magnitude |t|.
    fn ulp_at(t: f64) -> f64 {
        let m = (t.abs() as f32).min(f32::MAX);
        (f32::from_bits(m.to_bits() + 1) - m) as f64
    }

    /// f32 ULPs between `got` and the f64 value `t`, measured at t's binade.
    fn ulps(got: f32, t: f64) -> f64 {
        if got.is_nan() || t.is_nan() {
            return if got.is_nan() && t.is_nan() { 0.0 } else { f64::INFINITY };
        }
        if got as f64 == t || got == t as f32 {
            return 0.0;
        }
        (got as f64 - t).abs() / ulp_at(t)
    }

    /// Every f32 binade from denormal to near-max, both signs, plus a dense
    /// grid across exp's whole finite-to-overflow range.
    fn wide_vals() -> Vec<f32> {
        let mut v = special_vals(5, 64);
        let r = lcg_fill(99, 4 * 280);
        for (k, e) in (-149i32..128).enumerate() {
            for j in 0..4 {
                let m = 1.0 + (r[4 * k + j] + 2.0) / 4.0;
                let x = m * 2f32.powi(e.max(-126)) * 2f32.powi((e + 126).min(0));
                v.extend([x, -x]);
            }
        }
        v.extend((0..20001).map(|i| -110.0 + i as f32 * 0.0105));
        v.extend([88.72, 88.73, -87.33, -103.9, -104.0, -105.0, 0.625, -0.625, 0.624_999_94]);
        v
    }

    /// Tolerance contract for the vectorized kinds (see avx2.rs): every lane
    /// at most 4 ULP further from the f64 formula than CpuBackend's result.
    #[test]
    fn unary_tolerance_approx_kinds() {
        let x = wide_vals();
        for &kind in &APPROX {
            let check = |xs: &[f32], got: &[f32], ctx: &str| {
                let cpu = CpuBackend.unary(kind, xs);
                let (mut worst, mut excess, mut rel) = (0f64, 0f64, 0f64);
                for ((&v, &f), &c) in xs.iter().zip(got).zip(&cpu) {
                    let t = truth(kind, v);
                    let (ef, ec) = (ulps(f, t), ulps(c, t));
                    assert!(ef <= ec + 4.0, "{kind:?} {ctx} x={v:e}: fast {f:e} ({ef} ulp) cpu {c:e} ({ec} ulp) truth {t:e}");
                    excess = excess.max(ef - ec);
                    if ec <= 4.0 && t.abs() >= f32::MIN_POSITIVE as f64 && t.abs() <= f32::MAX as f64 {
                        worst = worst.max(ef);
                        rel = rel.max((f as f64 - c as f64).abs() / t.abs());
                    }
                }
                (worst, excess, rel)
            };
            for &len in &LENGTHS {
                let xs = special_vals(len as u64 * 7 + 3, len);
                check(&xs, &FastCpuBackend.unary(kind, &xs), &format!("len={len}"));
            }
            check(&x, &unary_serial(kind, &x), "wide serial");
            let (worst, excess, rel) = check(&x, &FastCpuBackend.unary(kind, &x), "wide");
            eprintln!("{kind:?}: at most {excess:.2} ulp further from the f64 formula than CpuBackend; where CpuBackend is within 4 ulp and normal: max {worst:.2} ulp, max |fast-cpu|/|t| {rel:.2e}");
        }
    }

    #[test]
    fn sum_and_sum_dim_match_reference_bitwise() {
        let lens = [0usize, 1, 7, 8, 9, 127, 128, 129, 255, 256, 257, 1000, 4095, 4096, 4097, 32 * 128 + 1, 100_003, (1 << 21) - 1, (1 << 21) + 5, (1 << 22) + 3];
        for &len in &lens {
            let x = lcg_fill(len as u64 + 17, len);
            assert_bitwise(&[FastCpuBackend.sum(&x)], &[CpuBackend.sum(&x)], &format!("sum len={len}"));
        }
        let specials = special_vals(3, 300);
        assert!(FastCpuBackend.sum(&specials).is_nan() && CpuBackend.sum(&specials).is_nan());
        let shapes: [&[usize]; 12] = [&[5], &[3, 1], &[1, 300], &[7, 129, 13], &[2, 3, 4, 5], &[1025, 9], &[9, 1025], &[4, 300, 3], &[256, 1024], &[3, 0, 4], &[2049, 1027], &[3, 700_001]];
        for shape in shapes {
            let x = lcg_fill(shape.len() as u64 * 31, shape.iter().product());
            for dim in 0..shape.len() {
                let ctx = format!("sum_dim shape={shape:?} dim={dim}");
                assert_bitwise(&FastCpuBackend.sum_dim(&x, shape, dim), &CpuBackend.sum_dim(&x, shape, dim), &ctx);
            }
        }
    }

    fn softmax_truth(row: &[f32], log: bool) -> Vec<f64> {
        let m = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
        let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
        row.iter().map(|&v| if log { v as f64 - m - s.ln() } else { (v as f64 - m).exp() / s }).collect()
    }

    /// Same contract as the unary kinds, in units of 2^-23 * |t| (one ULP at
    /// the bottom of t's binade, so 4 units = 4.8e-7 relative): a one-ULP
    /// exp difference just above a power of two would otherwise count double
    /// after the divide. log_softmax measures at the row's largest |output|,
    /// since x - lse cancels near the row max.
    #[test]
    fn softmax_tolerance() {
        for (rows, cols) in [(0, 5), (3, 0), (1, 1), (4, 7), (9, 33), (1024, 1024), (2, 5000), (300, 129)] {
            let mut x = lcg_fill(rows as u64 * 7 + cols as u64, rows * cols);
            for v in x.iter_mut().skip(3).step_by(17) {
                *v *= 20.0;
            }
            for log in [false, true] {
                let (fast, cpu) = if log {
                    (FastCpuBackend.log_softmax(&x, rows, cols), CpuBackend.log_softmax(&x, rows, cols))
                } else {
                    (FastCpuBackend.softmax(&x, rows, cols), CpuBackend.softmax(&x, rows, cols))
                };
                assert_eq!(fast.len(), rows * cols);
                let (mut worst, mut excess) = (0f64, 0f64);
                for r in 0..rows {
                    let t = softmax_truth(&x[r * cols..(r + 1) * cols], log);
                    let unit = ulp_at(t.iter().fold(0f64, |m, v| m.max(v.abs())));
                    for k in 0..cols {
                        let i = r * cols + k;
                        let unit = if log { unit } else { (t[k] * f32::EPSILON as f64).max(f32::from_bits(1) as f64) };
                        let at = |v: f32| (v as f64 - t[k]).abs() / unit;
                        let (ef, ec) = (at(fast[i]), at(cpu[i]));
                        assert!(ef <= ec + 4.0, "log={log} {rows}x{cols} [{r},{k}]: fast {} ({ef}) cpu {} ({ec})", fast[i], cpu[i]);
                        worst = worst.max(ef);
                        excess = excess.max(ef - ec);
                    }
                }
                if rows * cols > 0 {
                    eprintln!("softmax log={log} {rows}x{cols}: max {worst:.2} ulp from f64, at most {excess:.2} ulp further than CpuBackend");
                }
            }
        }
        let specials = [1.0, f32::NAN, 2.0, f32::INFINITY, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, -1.0, 3.0, 0.5, -2.0];
        for log in [false, true] {
            for cols in [1, 2, 3, 4, 12] {
                let rows = specials.len() / cols;
                let (fast, cpu) = if log {
                    (FastCpuBackend.log_softmax(&specials, rows, cols), CpuBackend.log_softmax(&specials, rows, cols))
                } else {
                    (FastCpuBackend.softmax(&specials, rows, cols), CpuBackend.softmax(&specials, rows, cols))
                };
                for (i, (f, c)) in fast.iter().zip(&cpu).enumerate() {
                    assert!(f.is_nan() == c.is_nan() && (f.is_nan() || f == c || (f - c).abs() <= 1e-6 * c.abs().max(1.0)), "specials log={log} cols={cols} [{i}]: {f} vs {c}");
                }
            }
        }
    }

    #[test]
    fn tensor_ops_route_through_backend() {
        let _registry = crate::registry_tests::lock();
        crate::install_backend();
        let x = Tensor::from_vec(lcg_fill(8, 6 * 40), &[6, 40]).unwrap();
        let v = x.to_vec();
        assert_eq!(x.sum().to_vec(), vec![CpuBackend.sum(&v)]);
        assert_eq!(x.sum_dim(0, false).unwrap().to_vec(), CpuBackend.sum_dim(&v, &[6, 40], 0));
        assert_eq!(x.softmax(1).unwrap().to_vec(), FastCpuBackend.softmax(&v, 6, 40));
        assert_eq!(x.log_softmax(1).unwrap().to_vec(), FastCpuBackend.log_softmax(&v, 6, 40));
        assert_eq!(x.exp().to_vec(), FastCpuBackend.unary(UnaryKind::Exp, &v));
    }

    #[test]
    fn binary_parity_all_kinds() {
        let kinds = [BinaryKind::Add, BinaryKind::Sub, BinaryKind::Mul, BinaryKind::Div];
        for (ki, &kind) in kinds.iter().enumerate() {
            for &len in &LENGTHS {
                let a = special_vals(len as u64 * 11 + ki as u64, len);
                let b = special_vals(len as u64 * 13 + ki as u64 + 1, len);
                let want = CpuBackend.binary(kind, &a, &b);
                let ctx = format!("{kind:?} len={len}");
                assert_bitwise(&FastCpuBackend.binary(kind, &a, &b), &want, &ctx);
                assert_bitwise(&binary_serial(kind, &a, &b), &want, &format!("{ctx} (serial)"));
            }
        }
    }

    #[test]
    fn install_backend_routes_tensor_ops() {
        let _registry = crate::registry_tests::lock();
        crate::install_backend();
        let x = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 1.5, 3.0], &[5]).unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        assert_eq!(x.relu().to_vec(), vec![0.0, 0.0, 0.0, 1.5, 3.0]);
        assert_eq!(x.add(&y).unwrap().to_vec(), vec![-1.0, 1.5, 3.0, 5.5, 8.0]);

    }

    // (m,k,n) mixes covering {1,5,17,64,128}, including m=1/n=1 degenerates
    // and every dim taking a turn at 1, mirroring op_bmm.rs's dispatch-parity
    // grid one crate over.
    const MATMUL_BATCH_DIMS: [(usize, usize, usize); 19] = [
        (1, 1, 1),
        (5, 5, 5),
        (17, 17, 17),
        (64, 64, 64),
        (128, 128, 128),
        (1, 5, 17),
        (5, 1, 17),
        (5, 17, 1),
        (1, 64, 128),
        (64, 1, 128),
        (64, 128, 1),
        (17, 64, 5),
        (64, 5, 17),
        (5, 64, 17),
        (1, 1, 128),
        (1, 128, 1),
        (128, 1, 1),
        (1, 17, 5),
        (128, 5, 17),
    ];

    // The installed CpuBackend callback and FastCpuBackend BMM both use
    // conservative arithmetic; the default wrapper calls it once per slab.
    #[test]
    fn matmul_batch_matches_default_bitwise() {
        let _registry = crate::registry_tests::lock();
        crate::install();
        for batch in [1usize, 3, 16] {
            for (i, &(m, k, n)) in MATMUL_BATCH_DIMS.iter().enumerate() {
                let a = lcg_fill(1000 + i as u64 + batch as u64 * 97, batch * m * k);
                let b = lcg_fill(2000 + i as u64 + batch as u64 * 97, batch * k * n);
                let seeds = [1000 + i as u64 + batch as u64 * 97, 2000 + i as u64 + batch as u64 * 97];
                let mut evidence = diagnostics::Capture::new(&a, &b, [batch, m, k, n], seeds, 0, None);
                let want = CpuBackend.matmul_batch(&a, &b, batch, m, k, n);
                evidence.before_fast(&a, &b, &want);
                let got = FastCpuBackend.matmul_batch(&a, &b, batch, m, k, n);
                evidence.on_failure(&a, &b, &got, &want, "matmul_batch_matches_default_bitwise");
                assert_bitwise(&got, &want, &format!("batch={batch} m={m} k={k} n={n}"));
            }
        }
    }

    #[test]
    fn matmul_batch_is_deterministic() {
        // Conservative BMM is serial even at this formerly threaded shape.
        // Two runs must still agree bitwise.
        let (batch, m, k, n) = (16usize, 128usize, 128usize, 128usize);
        let a = lcg_fill(11, batch * m * k);
        let b = lcg_fill(13, batch * k * n);
        let mut evidence = diagnostics::Capture::new(&a, &b, [batch, m, k, n], [11, 13], 0, None);
        let r1 = FastCpuBackend.matmul_batch(&a, &b, batch, m, k, n);
        evidence.before_fast(&a, &b, &r1);
        let r2 = FastCpuBackend.matmul_batch(&a, &b, batch, m, k, n);
        evidence.on_failure(&a, &b, &r2, &r1, "matmul_batch determinism");
        assert_bitwise(&r2, &r1, "matmul_batch determinism");
    }
}
