//! Optimized f32 CPU backend for ferro-core, registered through the kernel
//! dispatch seam without touching core. Pure std, no external deps.
//! Matmul and batched matmul run the packed BLIS-style SGEMM in gemm.rs
//! (runtime-dispatched AVX-512 / AVX2+FMA / scalar micro-kernels, threaded
//! over row or column blocks on the worker pool above a work threshold). See
//! gemm.rs for the numerical contract and verification/cpu-gemm-incidents/
//! for why the former conservative BMM containment was retired.

#[cfg(target_arch = "x86_64")]
mod avx2;
pub mod check;
pub mod elementwise;
pub mod gemm;
mod workers;
#[cfg(test)]
mod gemm_tests;
#[cfg(test)]
mod registry_tests;

/// Row-major (m,k) @ (k,n) -> (m,n), threaded above a work threshold.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    matmul_with_threads(a, b, m, k, n, gemm::auto_threads(m.saturating_mul(k).saturating_mul(n)))
}

/// `matmul` with an explicit thread count (benchmarking knob; results are
/// bitwise identical for every count).
pub fn matmul_with_threads(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, threads: usize) -> Vec<f32> {
    matmul_batch_with_threads(a, b, 1, m, k, n, threads)
}

/// Row-major (batch,m,k) @ (batch,k,n) -> (batch,m,n).
pub fn matmul_batch(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let work = [batch, m, k, n].iter().fold(1usize, |w, &d| w.saturating_mul(d));
    matmul_batch_with_threads(a, b, batch, m, k, n, gemm::auto_threads(work))
}

fn matmul_batch_with_threads(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize, threads: usize) -> Vec<f32> {
    if batch == 0 || m == 0 || n == 0 { return Vec::new(); }
    let rows = batch.checked_mul(m).expect("BMM row count overflow");
    let alen = rows.checked_mul(k).expect("BMM input size overflow");
    let blen = k.checked_mul(n).and_then(|s| s.checked_mul(batch)).expect("BMM input size overflow");
    let olen = rows.checked_mul(n).expect("BMM output size overflow");
    assert!(a.len() >= alen, "BMM A buffer too short");
    assert!(b.len() >= blen, "BMM B buffer too short");
    let mut out = ferro_core::pool::take_uninit(olen);
    gemm::sgemm_batch(a, b, &mut out, batch, m, k, n, threads);
    check::verify(a, b, &out, batch, m, k, n);
    out
}

/// Register the fast matmul kernel for CpuBackend (its default BMM calls it
/// once per slab).
pub fn install() {
    ferro_core::dispatch::set_matmul_kernel(matmul);
}

/// Register the vectorized/threaded elementwise backend process-wide for
/// Device::Cpu.
pub fn install_backend() {
    ferro_core::register_backend(ferro_core::Device::Cpu, std::sync::Arc::new(elementwise::FastCpuBackend));
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
    fn zero_dims_produce_empty_or_zero_outputs() {
        // n == 0 with threads > 1 used to hit chunks_mut(0) and panic.
        for (m, k, n) in [(0, 8, 8), (8, 0, 8), (8, 8, 0), (0, 0, 0)] {
            let a = lcg_fill(1, m * k);
            let b = lcg_fill(2, k * n);
            assert_eq!(matmul(&a, &b, m, k, n), naive_matmul(&a, &b, m, k, n));
            assert_eq!(matmul_with_threads(&a, &b, m, k, n, 4), naive_matmul(&a, &b, m, k, n));
            let batched = matmul_batch(&a, &b, 1, m, k, n);
            assert_eq!(batched, naive_matmul(&a, &b, m, k, n));
        }
        assert!(matmul_batch(&[], &[], 0, 4, 4, 4).is_empty());
    }

    #[test]
    fn matches_naive_threaded_path() {
        // m*k*n > GRAIN: exercises the std::thread::scope split.
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
        // Same remainder stress, sized to also exercise the multi-threaded
        // split with packed edge tiles.
        for (m, k, n) in [(129, 257, 65), (127, 300, 129), (65, 256, 127)] {
            check_shape(m, k, n);
        }
    }

    #[test]
    fn install_routes_tensor_matmul() {
        let _registry = crate::registry_tests::lock();
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
