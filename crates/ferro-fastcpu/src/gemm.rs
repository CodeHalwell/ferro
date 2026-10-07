//! Packed BLIS-style SGEMM, C = A @ B, all row-major, C overwritten.
//!
//! Loop nest: jc (NC cols) -> pc (KC) -> pack B -> ic (MC rows) -> pack A ->
//! jr (NR) -> ir (MR) -> register-blocked micro-kernel. A is packed as plain
//! rows (a memcpy per row) and broadcast at a fixed row stride; B is packed
//! into NR-wide panels. M edges run an R-row kernel instantiation; N edges
//! run on a zero-padded panel into a stack tile and copy back the valid cols.
//!
//! Numerical contract: every output is ONE multiply-add chain over p in
//! ascending order starting from +0: fused (one rounding per step, exactly
//! `f32::mul_add`) on the AVX2 and AVX-512 paths, separate multiply then add
//! on the scalar fallback. Blocking never splits that chain - for pc > 0 the
//! kernel reloads the partial C into its accumulators - and threads own
//! disjoint outputs, so results are bitwise independent of thread count,
//! block sizes, edge tiles and batch decomposition, and the two SIMD paths
//! agree bitwise with each other.
//!
//! Every SIMD entry is a #[target_feature] function reached only through
//! `Isa::detect`, which checks the CPU at runtime; nothing here is compiled
//! for a feature level above the crate's baseline target.

use crate::workers;
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    Avx512,
    Avx2,
    Scalar,
}

impl Isa {
    /// Best ISA this CPU supports. Runtime-detected, never assumed.
    pub fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Isa::Avx512;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Isa::Avx2;
            }
        }
        Isa::Scalar
    }

    /// Whether this CPU can run `self` (callers forcing an ISA must check).
    pub fn supported(self) -> bool {
        match self {
            Isa::Scalar => true,
            Isa::Avx2 => Isa::detect() != Isa::Scalar,
            Isa::Avx512 => Isa::detect() == Isa::Avx512,
        }
    }

    fn params(self) -> Params {
        match self {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => Params { mr: 14, nr: 32, mc: 196, nc: 3072, kernels: &K512 },
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => Params { mr: 6, nr: 16, mc: 96, nc: 3072, kernels: &K256 },
            _ => Params { mr: 4, nr: 16, mc: 64, nc: 3072, kernels: &KSCALAR },
        }
    }
}

/// `kernel(kc, pa, pb, c, ldc, load)` for an R-row kernel: the R x NR tile at
/// `c` (row stride ldc) = (load ? C : 0) + packed A rows (stride KA) times the
/// packed kc x NR B panel. R ranges over 1..=MR, so M edges need no padding.
type Kernel = unsafe fn(usize, *const f32, *const f32, *mut f32, usize, bool);

struct Params {
    mr: usize,
    nr: usize,
    mc: usize,
    nc: usize,
    kernels: &'static [Kernel],
}

/// Nominal K block. Blocks are balanced up to KC + KC/4 deep, so a packed A
/// row holds at most KA floats; KA is a compile-time row stride, which keeps
/// every A broadcast a constant-displacement load.
const KC: usize = 256;
const KA: usize = KC + KC / 4;
const MAX_TILE: usize = 14 * 32;

/// Below this many multiply-adds per pool task, waking a worker costs more
/// than it saves.
const GRAIN: usize = 1 << 20;

/// Task count for `work` multiply-adds: 1 for small problems.
pub fn auto_threads(work: usize) -> usize {
    (work / GRAIN).clamp(1, workers::threads())
}

/// C (m x n) = A (m x k) @ B (k x n) on `threads` threads with the
/// detected ISA.
pub fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, threads: usize) {
    sgemm_isa(Isa::detect(), a, b, c, m, k, n, threads)
}

/// `sgemm` with an explicit ISA. Panics if the CPU does not support it.
pub fn sgemm_isa(isa: Isa, a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, threads: usize) {
    assert!(isa.supported(), "{isa:?} not supported by this CPU");
    let (mk, kn, mn) = (m.checked_mul(k), k.checked_mul(n), m.checked_mul(n));
    let (Some(mk), Some(kn), Some(mn)) = (mk, kn, mn) else { panic!("sgemm size overflow") };
    assert!(a.len() >= mk && b.len() >= kn && c.len() >= mn, "sgemm buffer too short");
    if mn == 0 {
        return;
    }
    let c = &mut c[..mn];
    if k == 0 {
        c.fill(0.0);
        return;
    }
    let p = isa.params();
    let threads = threads.max(1);
    // Split rows while every task still gets several MR slivers; otherwise
    // split columns (one private output buffer per task, copied back). Tasks
    // run on the persistent pool; each owns its part behind an uncontended
    // Mutex, which is how a shared `Fn` task hands out disjoint `&mut`s.
    let row_threads = threads.min(m.div_ceil(p.mr * 4)).max(1);
    if threads == 1 || row_threads == threads || n < p.nr * 2 {
        let rows = m.div_ceil(row_threads).next_multiple_of(p.mr);
        if rows >= m {
            serial(&p, a, k, b, n, c, n, m, k, n);
            return;
        }
        let parts: Vec<Mutex<&mut [f32]>> = c.chunks_mut(rows * n).map(Mutex::new).collect();
        workers::run(parts.len(), &|t| {
            let mut part = parts[t].lock().unwrap_or_else(|e| e.into_inner());
            let r = part.len() / n;
            serial(&p, &a[t * rows * k..], k, b, n, &mut part, n, r, k, n);
        });
        return;
    }
    let cols = n.div_ceil(threads).next_multiple_of(p.nr);
    let ranges: Vec<(usize, usize)> = (0..n).step_by(cols).map(|j| (j, cols.min(n - j))).collect();
    let bufs: Vec<Mutex<Vec<f32>>> = ranges.iter().map(|&(_, w)| Mutex::new(vec![0.0; m * w])).collect();
    workers::run(ranges.len(), &|t| {
        let (j, w) = ranges[t];
        let mut buf = bufs[t].lock().unwrap_or_else(|e| e.into_inner());
        serial(&p, a, k, &b[j..], n, &mut buf, w, m, k, w);
    });
    for (&(j, w), buf) in ranges.iter().zip(bufs) {
        let buf = buf.into_inner().unwrap_or_else(|e| e.into_inner());
        for (dst, src) in c.chunks_exact_mut(n).zip(buf.chunks_exact(w)) {
            dst[j..j + w].copy_from_slice(src);
        }
    }
}

/// Batched GEMM over contiguous slabs. Whole batch elements go to threads
/// when there are enough of them; otherwise each element is threaded.
pub fn sgemm_batch(a: &[f32], b: &[f32], c: &mut [f32], batch: usize, m: usize, k: usize, n: usize, threads: usize) {
    let (mk, kn, mn) = (m * k, k * n, m * n);
    if batch == 0 || mn == 0 {
        return;
    }
    if batch < threads.max(2) || threads == 1 {
        for (bi, cs) in c.chunks_exact_mut(mn).take(batch).enumerate() {
            sgemm(&a[bi * mk..], &b[bi * kn..], cs, m, k, n, if batch < threads { threads } else { 1 });
        }
        return;
    }
    let per = batch.div_ceil(threads);
    let parts: Vec<Mutex<&mut [f32]>> = c[..batch * mn].chunks_mut(per * mn).map(Mutex::new).collect();
    workers::run(parts.len(), &|t| {
        let mut part = parts[t].lock().unwrap_or_else(|e| e.into_inner());
        for (i, cb) in part.chunks_exact_mut(mn).enumerate() {
            let bi = t * per + i;
            sgemm(&a[bi * mk..], &b[bi * kn..], cb, m, k, n, 1);
        }
    });
}

/// Single-threaded blocked GEMM on a sub-problem: A rows with stride lda,
/// B rows with stride ldb (B may be a column window), C rows with stride ldc.
#[allow(clippy::too_many_arguments)]
fn serial(p: &Params, a: &[f32], lda: usize, b: &[f32], ldb: usize, c: &mut [f32], ldc: usize, m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(a.len() >= (m - 1) * lda + k && b.len() >= (k - 1) * ldb + n && c.len() >= (m - 1) * ldc + n);
    let (mr, nr) = (p.mr, p.nr);
    let nc_max = p.nc.min(n.next_multiple_of(nr));
    let mc_max = p.mc.min(m.next_multiple_of(mr));
    // Balanced K blocks (a 288-deep problem is one 288 block, not 256 + 32),
    // so no block is too shallow to amortize its C load/store.
    let kc_max = k.div_ceil(k.div_ceil(KA));
    SCRATCH.with_borrow_mut(|buf| {
        if buf.len() < nc_max * kc_max + mc_max * KA {
            buf.resize(nc_max * kc_max + mc_max * KA, 0.0);
        }
        let (pb, pa) = buf.split_at_mut(nc_max * kc_max);
        blocked(p, a, lda, b, ldb, c, ldc, m, k, n, kc_max, pa, pb);
    });
}

thread_local! {
    /// Packing buffers, reused across calls on the same thread. Packing
    /// writes every element the kernel then reads, so stale contents from an
    /// earlier call are never consumed (tests poison it to prove that).
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn poison_scratch() {
    SCRATCH.with_borrow_mut(|buf| buf.iter_mut().for_each(|v| *v = f32::NAN));
}

#[allow(clippy::too_many_arguments)]
fn blocked(p: &Params, a: &[f32], lda: usize, b: &[f32], ldb: usize, c: &mut [f32], ldc: usize, m: usize, k: usize, n: usize, kcb: usize, pa: &mut [f32], pb: &mut [f32]) {
    let (mr, nr) = (p.mr, p.nr);
    let mut tile = [0.0f32; MAX_TILE];
    for jc in (0..n).step_by(p.nc) {
        let nc = p.nc.min(n - jc);
        for pc in (0..k).step_by(kcb) {
            let kc = kcb.min(k - pc);
            pack_b(pb, &b[pc * ldb + jc..], ldb, kc, nc, nr);
            for ic in (0..m).step_by(p.mc) {
                let mc = p.mc.min(m - ic);
                for (r, row) in pa.chunks_exact_mut(KA).take(mc).enumerate() {
                    row[..kc].copy_from_slice(&a[(ic + r) * lda + pc..][..kc]);
                }
                for jr in (0..nc).step_by(nr) {
                    let nw = nr.min(nc - jr);
                    let bp = pb[jr * kc..][..kc * nr].as_ptr();
                    for ir in (0..mc).step_by(mr) {
                        let mw = mr.min(mc - ir);
                        let ap = pa[ir * KA..][..(mw - 1) * KA + kc].as_ptr();
                        let kernel = p.kernels[mw - 1];
                        let off = (ic + ir) * ldc + jc + jr;
                        if nw == nr {
                            let dst = &mut c[off..off + (mw - 1) * ldc + nr];
                            // SAFETY: the mw-row kernel reads mw A rows of kc floats at
                            // stride KA and a kc x NR B panel, and touches mw rows of NR
                            // floats at stride ldc - all inside the slices above. The
                            // ISA was checked by sgemm_isa.
                            unsafe { kernel(kc, ap, bp, dst.as_mut_ptr(), ldc, pc > 0) };
                        } else {
                            let t = &mut tile[..mw * nr];
                            if pc > 0 {
                                for r in 0..mw {
                                    t[r * nr..r * nr + nw].copy_from_slice(&c[off + r * ldc..][..nw]);
                                }
                            }
                            // SAFETY: as above, with the stack tile (ldc = nr) as C;
                            // the panel's columns past nw are zero padding.
                            unsafe { kernel(kc, ap, bp, t.as_mut_ptr(), nr, pc > 0) };
                            for r in 0..mw {
                                c[off + r * ldc..][..nw].copy_from_slice(&t[r * nr..r * nr + nw]);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// dst[q][p][j] = B[p][q*nr + j] for the kc x nc window, tail cols zeroed.
fn pack_b(dst: &mut [f32], b: &[f32], ldb: usize, kc: usize, nc: usize, nr: usize) {
    for (q, panel) in dst.chunks_exact_mut(kc * nr).take(nc.div_ceil(nr)).enumerate() {
        let j0 = q * nr;
        let w = nr.min(nc - j0);
        for (p, row) in panel.chunks_exact_mut(nr).enumerate() {
            row[..w].copy_from_slice(&b[p * ldb + j0..][..w]);
            row[w..].fill(0.0);
        }
    }
}

macro_rules! table {
    ($k:ident) => {
        [$k::<1>, $k::<2>, $k::<3>, $k::<4>, $k::<5>, $k::<6>, $k::<7>, $k::<8>, $k::<9>, $k::<10>, $k::<11>, $k::<12>, $k::<13>, $k::<14>]
    };
}

static KSCALAR: [Kernel; 4] = [scalar_kernel::<1>, scalar_kernel::<2>, scalar_kernel::<3>, scalar_kernel::<4>];
#[cfg(target_arch = "x86_64")]
static K256: [Kernel; 6] = [avx2_kernel::<1>, avx2_kernel::<2>, avx2_kernel::<3>, avx2_kernel::<4>, avx2_kernel::<5>, avx2_kernel::<6>];
#[cfg(target_arch = "x86_64")]
static K512: [Kernel; 14] = table!(avx512_kernel);

/// Portable fallback: separate multiply and add (no FMA available).
unsafe fn scalar_kernel<const R: usize>(kc: usize, pa: *const f32, pb: *const f32, c: *mut f32, ldc: usize, load: bool) {
    const NR: usize = 16;
    let mut acc = [[0.0f32; NR]; R];
    if load {
        for (r, row) in acc.iter_mut().enumerate() {
            row.copy_from_slice(std::slice::from_raw_parts(c.add(r * ldc), NR));
        }
    }
    let pb = std::slice::from_raw_parts(pb, kc * NR);
    for (p, bv) in pb.chunks_exact(NR).enumerate() {
        for (r, row) in acc.iter_mut().enumerate() {
            let av = *pa.add(r * KA + p);
            for j in 0..NR {
                row[j] += av * bv[j];
            }
        }
    }
    for (r, row) in acc.iter().enumerate() {
        std::slice::from_raw_parts_mut(c.add(r * ldc), NR).copy_from_slice(row);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_kernel<const R: usize>(kc: usize, pa: *const f32, pb: *const f32, c: *mut f32, ldc: usize, load: bool) {
    use std::arch::x86_64::*;
    let mut c0 = [_mm256_setzero_ps(); R];
    let mut c1 = [_mm256_setzero_ps(); R];
    if load {
        for r in 0..R {
            c0[r] = _mm256_loadu_ps(c.add(r * ldc));
            c1[r] = _mm256_loadu_ps(c.add(r * ldc + 8));
        }
    }
    let (mut a, mut b) = (pa, pb);
    for _ in 0..kc {
        _mm_prefetch::<_MM_HINT_T0>(b.wrapping_add(128) as *const i8);
        let b0 = _mm256_loadu_ps(b);
        let b1 = _mm256_loadu_ps(b.add(8));
        for r in 0..R {
            let av = _mm256_broadcast_ss(&*a.add(r * KA));
            c0[r] = _mm256_fmadd_ps(av, b0, c0[r]);
            c1[r] = _mm256_fmadd_ps(av, b1, c1[r]);
        }
        a = a.add(1);
        b = b.add(16);
    }
    for r in 0..R {
        _mm256_storeu_ps(c.add(r * ldc), c0[r]);
        _mm256_storeu_ps(c.add(r * ldc + 8), c1[r]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_kernel<const R: usize>(kc: usize, pa: *const f32, pb: *const f32, c: *mut f32, ldc: usize, load: bool) {
    use std::arch::x86_64::*;
    let mut c0 = [_mm512_setzero_ps(); R];
    let mut c1 = [_mm512_setzero_ps(); R];
    if load {
        for r in 0..R {
            c0[r] = _mm512_loadu_ps(c.add(r * ldc));
            c1[r] = _mm512_loadu_ps(c.add(r * ldc + 16));
        }
    }
    let (mut a, mut b) = (pa, pb);
    for _ in 0..kc {
        // B streams from L2 when the panel outgrows L1: fetch 8 rows ahead
        // (a hint, so running past the panel end is harmless).
        _mm_prefetch::<_MM_HINT_T0>(b.wrapping_add(256) as *const i8);
        _mm_prefetch::<_MM_HINT_T0>(b.wrapping_add(272) as *const i8);
        let b0 = _mm512_loadu_ps(b);
        let b1 = _mm512_loadu_ps(b.add(16));
        for r in 0..R {
            let av = _mm512_set1_ps(*a.add(r * KA));
            c0[r] = _mm512_fmadd_ps(av, b0, c0[r]);
            c1[r] = _mm512_fmadd_ps(av, b1, c1[r]);
        }
        a = a.add(1);
        b = b.add(32);
    }
    for r in 0..R {
        _mm512_storeu_ps(c.add(r * ldc), c0[r]);
        _mm512_storeu_ps(c.add(r * ldc + 16), c1[r]);
    }
}
