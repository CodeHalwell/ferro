//! Vectorized/threaded elementwise CPU backend, registered through
//! `ferro_core::dispatch::register_backend` (see `install_backend` in
//! lib.rs). Every kind's formula is copied verbatim from `CpuBackend` - see
//! its comments for why each is written the way it is (NaN-propagating relu,
//! max/min-chain clamp, etc.) - so output is bit-for-bit identical; only the
//! loop shape changes:
//! - kind is matched once, outside the loop, so each generated loop body is
//!   a single monomorphized formula rather than a per-element branch
//! - chunks_exact(8) so LLVM sees a fixed-width inner loop and can
//!   autovectorize the arithmetic-only kinds (add/mul/relu/clamp/...);
//!   transcendental kinds (exp/sigmoid/tanh/powf/log) still call the scalar
//!   libm routine per lane (a vectorized approximation would break bitwise
//!   parity) but pay no extra loop overhead for it
//! - runtime AVX2+FMA dispatch via #[target_feature], exactly like lib.rs's
//!   matmul: plain builds target baseline SSE2 (4-wide f32), so this is what
//!   gives the chunks_exact(8) loop a matching 8-wide ymm register
//! - std::thread::scope above PAR_THRESHOLD elements; elementwise ops are
//!   bandwidth-bound (docs/CAPABILITY.md 5.1), so threads mainly pay off
//!   once the working set stops fitting L3.
//!
//! PAR_THRESHOLD was picked by sweeping binary-op (3-buffer) sizes on the
//! 4-core/33MB-L3 dev machine: at 1<<20 elements (4 MiB/buffer, 12 MiB
//! combined - comfortably L3-resident) threading is a net LOSS (0.7-0.85x:
//! spawn/join overhead loses to a single core that already saturates
//! L3-resident bandwidth). The win shows up once the combined working set
//! passes L3 capacity: at 1<<21 elements (8 MiB/buffer, 24 MiB combined)
//! threading is consistently >1.4x over the single-threaded vectorized path
//! and keeps paying off out to 32M+. See bench_elementwise for the full
//! table this was measured from.

use std::thread;

use ferro_core::dispatch::{Backend, BinaryKind, UnaryKind};

const PAR_THRESHOLD: usize = 1 << 21;

pub struct FastCpuBackend;

impl Backend for FastCpuBackend {
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        unary_dispatch(kind, x, &mut out);
        out
    }

    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; a.len()];
        binary_dispatch(kind, a, b, &mut out);
        out
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        crate::matmul(a, b, m, k, n)
    }
}

/// Vectorized but forced single-threaded; exposed so bench_elementwise can
/// isolate the vectorization win from the threading win.
pub fn unary_serial(kind: UnaryKind, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    unary_chunk(kind, x, &mut out);
    out
}

pub fn binary_serial(kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; a.len()];
    binary_chunk(kind, a, b, &mut out);
    out
}

fn unary_dispatch(kind: UnaryKind, x: &[f32], out: &mut [f32]) {
    if x.len() < PAR_THRESHOLD {
        unary_chunk(kind, x, out);
        return;
    }
    let threads = thread::available_parallelism().map_or(1, |p| p.get()).min(x.len());
    let per = x.len().div_ceil(threads);
    thread::scope(|s| {
        for (xc, oc) in x.chunks(per).zip(out.chunks_mut(per)) {
            s.spawn(move || unary_chunk(kind, xc, oc));
        }
    });
}

fn binary_dispatch(kind: BinaryKind, a: &[f32], b: &[f32], out: &mut [f32]) {
    if a.len() < PAR_THRESHOLD {
        binary_chunk(kind, a, b, out);
        return;
    }
    let threads = thread::available_parallelism().map_or(1, |p| p.get()).min(a.len());
    let per = a.len().div_ceil(threads);
    thread::scope(|s| {
        for ((ac, bc), oc) in a.chunks(per).zip(b.chunks(per)).zip(out.chunks_mut(per)) {
            s.spawn(move || binary_chunk(kind, ac, bc, oc));
        }
    });
}

fn unary_chunk(kind: UnaryKind, x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: avx2 and fma were just detected at runtime.
        unsafe { unary_chunk_avx2(kind, x, out) };
        return;
    }
    unary_chunk_body(kind, x, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unary_chunk_avx2(kind: UnaryKind, x: &[f32], out: &mut [f32]) {
    unary_chunk_body(kind, x, out);
}

fn binary_chunk(kind: BinaryKind, a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: avx2 and fma were just detected at runtime.
        unsafe { binary_chunk_avx2(kind, a, b, out) };
        return;
    }
    binary_chunk_body(kind, a, b, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn binary_chunk_avx2(kind: BinaryKind, a: &[f32], b: &[f32], out: &mut [f32]) {
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
    use std::sync::{Arc, Mutex};

    use ferro_core::{CpuBackend, Device, Tensor};

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
            assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: mismatch at index {i}: {x} vs {y}");
        }
    }

    // The last two exercise the single-threaded-but-vectorized path (well
    // under PAR_THRESHOLD) and the multithreaded path (just over it), both
    // with a non-8-aligned remainder.
    const LENGTHS: [usize; 11] = [0, 1, 7, 8, 9, 15, 16, 17, 1023, (1 << 20) + 3, PAR_THRESHOLD + 3];

    #[test]
    fn unary_parity_all_kinds() {
        let kinds = [
            UnaryKind::Neg,
            UnaryKind::Relu,
            UnaryKind::Exp,
            UnaryKind::Sigmoid,
            UnaryKind::Tanh,
            UnaryKind::Sqrt,
            UnaryKind::Abs,
            UnaryKind::Log,
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

    // Registration is process-global state shared with every other test in
    // this binary; serialize on a poison-tolerant lock and restore
    // CpuBackend afterward so unrelated tests keep seeing the default.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn install_backend_routes_tensor_ops() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        crate::install_backend();
        let x = Tensor::from_vec(vec![-2.0, -0.5, 0.0, 1.5, 3.0], &[5]).unwrap();
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        assert_eq!(x.relu().to_vec(), vec![0.0, 0.0, 0.0, 1.5, 3.0]);
        assert_eq!(x.add(&y).unwrap().to_vec(), vec![-1.0, 1.5, 3.0, 5.5, 8.0]);
        ferro_core::register_backend(Device::Cpu, Arc::new(CpuBackend));
    }
}
