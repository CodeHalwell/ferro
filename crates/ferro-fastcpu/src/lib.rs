//! Optimized f32 CPU matmul backend for ferro-core, registered through the
//! kernel dispatch seam (`ferro_core::dispatch::set_matmul_kernel`) without
//! touching core. Pure std, no external deps. Full Goto/van de Geijn (BLIS)
//! blocking structure around a 6x16 register-blocked micro-kernel:
//! - B is packed into contiguous (KC x NR) panels once per k-block, before
//!   the row sweep, so the micro-kernel always reads sequential memory
//! - A is packed into contiguous (MC x KC) panels (MR-row slivers) once per
//!   (row-block, k-block) pair, so A loads never stride by k
//! - the MC loop is the L2 blocking level: the packed A panel for one
//!   (row-block, k-block) pair is swept over every n-tile before moving on,
//!   so it stays L2-resident across that whole sweep
//! - edge M/N tiles are zero-padded in the packed buffers so the
//!   micro-kernel always runs a full MR x NR update, with a masked
//!   writeback for the valid rows/cols - there is no scalar remainder path
//! - runtime AVX2+FMA dispatch via #[target_feature]; plain `cargo` builds
//!   target baseline SSE2, so this is what unlocks the wide FMA units
//! - std::thread::scope splits M across available_parallelism() for large
//!   problems; each thread packs its own A/B panels into buffers allocated
//!   once per thread call and reused across blocks (never shared)

use std::thread;

pub mod elementwise;

/// Micro-kernel tile: MR x NR accumulators = 12 ymm registers under AVX2,
/// leaving room for B loads and A broadcasts (tuned: beats 4x16/8x16/6x32).
const MR: usize = 6;
const NR: usize = 16;
/// K block: (KC x NR) packed B panel is 16KB, resident in L1 across the
/// inner micro-kernel sweep.
const KC: usize = 256;
/// M block: the (MC x KC) packed A panel is resident in L2 across the whole
/// n sweep for that block. Tuned by measuring 1024^3 and 2048^3 single- and
/// multi-threaded GFLOP/s across {48, 72, 96, 120, 144, 192} (all multiples
/// of MR, so a full block's last A tile never needs edge padding), plus the
/// skinny (2048, 64, 2048) shape: 48 won cleanly and consistently across the
/// whole grid, not just within noise - e.g. ~40 vs ~36-37 GFLOP/s at 1024^3
/// single-threaded, and >2x on the skinny shape where larger MC values waste
/// L2 traffic re-sweeping mostly-empty (mc x 64) panels. The smallest
/// candidate wins here because KC=256 already keeps the panel L1/L2-small;
/// a bigger MC just adds re-pack/re-read cost without a matching reuse win
/// at these problem sizes.
const MC: usize = 48;
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

/// Like `matmul` but with an explicit thread count in place of the
/// PAR_THRESHOLD/available_parallelism heuristic - a benchmarking knob (see
/// src/bin/bench.rs) for measuring single- vs multi-threaded throughput at
/// a fixed problem size. `threads <= 1` runs inline with no spawn, matching
/// `matmul`'s small-problem path.
pub fn matmul_with_threads(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, threads: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    let threads = threads.clamp(1, m.max(1));
    if threads == 1 {
        matmul_rows(a, b, &mut out, k, n, 0);
        return out;
    }
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

/// Packs the (kc x n) row-block of B starting at k-offset `pp` into `dst` as
/// `njtiles` contiguous (kc x NR) panels, zero-padding the last panel's
/// tail columns when n isn't a multiple of NR.
#[inline(always)]
fn pack_b(dst: &mut [f32], b: &[f32], n: usize, pp: usize, kc: usize, njtiles: usize) {
    for jt in 0..njtiles {
        let jj = jt * NR;
        let ncols = NR.min(n - jj);
        let base = jt * kc * NR;
        for p in 0..kc {
            let src = (pp + p) * n + jj;
            let dstp = base + p * NR;
            dst[dstp..dstp + ncols].copy_from_slice(&b[src..src + ncols]);
            if ncols < NR {
                dst[dstp + ncols..dstp + NR].fill(0.0);
            }
        }
    }
}

/// Packs the (mc x kc) block of A (global rows i0..i0+mc, k-offset `pp`)
/// into `dst` as `mtiles` contiguous (kc x MR) panels:
/// `dst[t][p*MR+r] = A[(i0+t*MR+r)*k+pp+p]`, zero-padding the last panel's
/// tail rows when mc isn't a multiple of MR. Reads a contiguous kc-run of
/// `a` per row, so the stride-k gather happens once here instead of on
/// every micro-kernel call.
#[inline(always)]
fn pack_a(dst: &mut [f32], a: &[f32], k: usize, i0: usize, pp: usize, kc: usize, mc: usize, mtiles: usize) {
    for t in 0..mtiles {
        let r0 = t * MR;
        let mr = MR.min(mc - r0);
        let base = t * kc * MR;
        for r in 0..mr {
            let arow = (i0 + r0 + r) * k + pp;
            for p in 0..kc {
                dst[base + p * MR + r] = a[arow + p];
            }
        }
        for r in mr..MR {
            for p in 0..kc {
                dst[base + p * MR + r] = 0.0;
            }
        }
    }
}

/// Rank-kc update of a full MRxNR register tile from packed panels. Both
/// panels are always full size (edge rows/cols zero-padded by the packer),
/// so this is the only micro-kernel shape - no scalar remainder variant.
#[inline(always)]
fn micro_packed(pa: &[f32], pb: &[f32], kc: usize, acc: &mut [[f32; NR]; MR]) {
    for p in 0..kc {
        let av: &[f32; MR] = (&pa[p * MR..p * MR + MR]).try_into().unwrap();
        let bv: &[f32; NR] = (&pb[p * NR..p * NR + NR]).try_into().unwrap();
        for r in 0..MR {
            let ar = av[r];
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
    let njtiles = n.div_ceil(NR);
    // Allocated once per thread call, reused (fully overwritten) across
    // every k-block/row-block iteration below - never shared across threads.
    let mut packed_b = vec![0f32; njtiles * KC * NR];
    let mut packed_a = vec![0f32; (MC / MR) * KC * MR];
    for pp in (0..k).step_by(KC) {
        let pend = (pp + KC).min(k);
        let kc = pend - pp;
        pack_b(&mut packed_b, b, n, pp, kc, njtiles);
        let mut r0 = 0;
        while r0 < rows {
            let mc = MC.min(rows - r0);
            let mtiles = mc.div_ceil(MR);
            pack_a(&mut packed_a, a, k, i0 + r0, pp, kc, mc, mtiles);
            for jt in 0..njtiles {
                let jj = jt * NR;
                let ncols = NR.min(n - jj);
                let pb = &packed_b[jt * kc * NR..][..kc * NR];
                let mut r = 0;
                while r < mc {
                    let mr = MR.min(mc - r);
                    let pa = &packed_a[(r / MR) * kc * MR..][..kc * MR];
                    let mut acc = [[0f32; NR]; MR];
                    if pp > 0 {
                        for rr in 0..mr {
                            acc[rr][..ncols].copy_from_slice(&out[(r0 + r + rr) * n + jj..][..ncols]);
                        }
                    }
                    micro_packed(pa, pb, kc, &mut acc);
                    for rr in 0..mr {
                        out[(r0 + r + rr) * n + jj..][..ncols].copy_from_slice(&acc[rr][..ncols]);
                    }
                    r += MR;
                }
            }
            r0 += mc;
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
    fn matches_naive_packed_remainders() {
        // Cross-product of m/n values landing on every side of an MR=6 /
        // NR=16 tile boundary, against k values landing on every side of a
        // KC=256 block boundary: stresses pack_a/pack_b zero-padding and
        // the masked writeback for edge tiles.
        let sizes = [1, 2, 5, 6, 7, 15, 16, 17, 31, 33, 63, 65, 127, 129];
        let ks = [1, 17, 255, 256, 257, 300];
        for &m in &sizes {
            for &n in &sizes {
                for &k in &ks {
                    check_shape(m, k, n);
                }
            }
        }
    }

    #[test]
    fn matches_naive_packed_remainders_threaded() {
        // Same remainder stress, sized above PAR_THRESHOLD to also exercise
        // the multi-threaded row split with packed edge tiles.
        for (m, k, n) in [(129, 257, 65), (127, 300, 129), (65, 256, 127)] {
            assert!(m * k * n > PAR_THRESHOLD);
            check_shape(m, k, n);
        }
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
