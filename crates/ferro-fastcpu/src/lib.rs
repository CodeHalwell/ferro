//! Optimized f32 CPU matmul backend for ferro-core, registered through the
//! kernel dispatch seam (`ferro_core::dispatch::set_matmul_kernel`) without
//! touching core. Pure std, no external deps. Techniques:
//! - register-blocked 6x16 micro-kernel: the output tile lives in registers
//!   across the whole k sweep, reading contiguous 16-float rows of B so the
//!   inner loop auto-vectorizes to FMAs
//! - cache blocking over k (KC=256) so the (KC x 16) B panel stays in L1
//!   while it is reused by every row block
//! - runtime AVX2+FMA dispatch via #[target_feature]; plain `cargo` builds
//!   target baseline SSE2, so this is what unlocks the wide FMA units
//! - std::thread::scope splitting M across available_parallelism() for
//!   large problems

use std::thread;

pub mod elementwise;

/// Micro-kernel tile: MR x NR accumulators = 12 ymm registers under AVX2,
/// leaving room for B loads and A broadcasts (tuned: beats 4x16/8x16/6x32).
const MR: usize = 6;
const NR: usize = 16;
/// K block: (KC x NR) B panel is 16KB, resident in L1 across the row sweep.
const KC: usize = 256;
/// Below this many multiply-adds (m*k*n), thread spawn overhead dominates.
const PAR_THRESHOLD: usize = 1 << 18;

/// Row-major (m,k) @ (k,n) -> (m,n).
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    if m * k * n <= PAR_THRESHOLD {
        matmul_rows(a, b, &mut out, k, n, 0);
        return out;
    }
    let threads = thread::available_parallelism().map_or(1, |p| p.get()).min(m);
    let rows_per = m.div_ceil(threads);
    thread::scope(|s| {
        for (t, chunk) in out.chunks_mut(rows_per * n).enumerate() {
            s.spawn(move || matmul_rows(a, b, chunk, k, n, t * rows_per));
        }
    });
    out
}

/// Register this kernel process-wide for all ferro-core CPU matmuls.
pub fn install() {
    ferro_core::dispatch::set_matmul_kernel(matmul);
}

/// Register the vectorized/threaded elementwise backend process-wide for
/// Device::Cpu.
pub fn install_backend() {
    ferro_core::register_backend(ferro_core::Device::Cpu, std::sync::Arc::new(elementwise::FastCpuBackend));
}

/// Computes global rows i0..i0+out.len()/n of A@B into `out`.
fn matmul_rows(a: &[f32], b: &[f32], out: &mut [f32], k: usize, n: usize, i0: usize) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: avx2 and fma were just detected at runtime.
        unsafe { matmul_rows_avx2(a, b, out, k, n, i0) };
        return;
    }
    matmul_rows_body(a, b, out, k, n, i0);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn matmul_rows_avx2(a: &[f32], b: &[f32], out: &mut [f32], k: usize, n: usize, i0: usize) {
    matmul_rows_body(a, b, out, k, n, i0);
}

/// Rank-KC update of an MRxNR register tile: for each p, broadcast one A
/// element per row against a contiguous NR-wide B row. The fixed-size arrays
/// keep the accumulators in registers and let LLVM emit packed FMAs.
#[inline(always)]
fn micro<const R: usize>(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    i: usize,
    jj: usize,
    p0: usize,
    p1: usize,
    acc: &mut [[f32; NR]; R],
) {
    for p in p0..p1 {
        let bv: &[f32; NR] = (&b[p * n + jj..p * n + jj + NR]).try_into().unwrap();
        for r in 0..R {
            let ar = a[(i + r) * k + p];
            for j in 0..NR {
                acc[r][j] += ar * bv[j];
            }
        }
    }
}

#[inline(always)]
fn matmul_rows_body(a: &[f32], b: &[f32], out: &mut [f32], k: usize, n: usize, i0: usize) {
    if n == 0 || out.is_empty() {
        return;
    }
    let rows = out.len() / n;
    let jful = n - n % NR;
    let rful = rows - rows % MR;
    for pp in (0..k).step_by(KC) {
        let pend = (pp + KC).min(k);
        for jj in (0..jful).step_by(NR) {
            let mut r = 0;
            while r < rful {
                let mut acc = [[0f32; NR]; MR];
                if pp > 0 {
                    for t in 0..MR {
                        acc[t].copy_from_slice(&out[(r + t) * n + jj..][..NR]);
                    }
                }
                micro::<MR>(a, b, k, n, i0 + r, jj, pp, pend, &mut acc);
                for t in 0..MR {
                    out[(r + t) * n + jj..][..NR].copy_from_slice(&acc[t]);
                }
                r += MR;
            }
            while r < rows {
                let mut acc = [[0f32; NR]; 1];
                if pp > 0 {
                    acc[0].copy_from_slice(&out[r * n + jj..][..NR]);
                }
                micro::<1>(a, b, k, n, i0 + r, jj, pp, pend, &mut acc);
                out[r * n + jj..][..NR].copy_from_slice(&acc[0]);
                r += 1;
            }
        }
        // n % NR tail columns: contiguous rank-1 updates (still vectorizable).
        if jful < n {
            for r in 0..rows {
                let orow = &mut out[r * n + jful..r * n + n];
                for p in pp..pend {
                    let ap = a[(i0 + r) * k + p];
                    let brow = &b[p * n + jful..p * n + n];
                    for j in 0..orow.len() {
                        orow[j] += ap * brow[j];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferro_core::dispatch::naive_matmul;
    use ferro_core::Tensor;

    fn lcg_fill(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    fn assert_close(fast: &[f32], reference: &[f32]) {
        assert_eq!(fast.len(), reference.len());
        for (i, (&x, &y)) in fast.iter().zip(reference).enumerate() {
            let tol = 1e-3 * y.abs().max(1.0);
            assert!((x - y).abs() <= tol, "mismatch at {i}: {x} vs {y}");
        }
    }

    fn check_shape(m: usize, k: usize, n: usize) {
        let a = lcg_fill(m as u64 * 31 + k as u64, m * k);
        let b = lcg_fill(n as u64 * 17 + 7, k * n);
        assert_close(&matmul(&a, &b, m, k, n), &naive_matmul(&a, &b, m, k, n));
    }

    #[test]
    fn matches_naive_across_shapes() {
        let shapes = [(1, 1, 1), (3, 5, 7), (64, 64, 64), (200, 300, 150), (17, 129, 33)];
        for (m, k, n) in shapes {
            check_shape(m, k, n);
        }
    }

    #[test]
    fn matches_naive_edge_dims() {
        // m=1 and n=1 stress the row-split and tile remainders.
        for (m, k, n) in [(1, 256, 256), (256, 256, 1), (1, 300, 1), (128, 1, 128)] {
            check_shape(m, k, n);
        }
    }

    #[test]
    fn matches_naive_threaded_path() {
        // m*k*n > PAR_THRESHOLD: exercises the std::thread::scope row split.
        check_shape(160, 96, 80);
    }

    #[test]
    fn install_routes_tensor_matmul() {
        let (m, k, n) = (12, 20, 9);
        let a = lcg_fill(3, m * k);
        let b = lcg_fill(4, k * n);
        let expected = naive_matmul(&a, &b, m, k, n);
        let ta = Tensor::from_vec(a, &[m, k]).unwrap();
        let tb = Tensor::from_vec(b, &[k, n]).unwrap();
        install();
        assert_close(&ta.matmul(&tb).unwrap().to_vec(), &expected);
    }
}
