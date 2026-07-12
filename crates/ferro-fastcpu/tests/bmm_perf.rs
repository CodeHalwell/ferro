//! Ignored timing test: bmm forward throughput naive vs. the swappable
//! single-matmul kernel (bmm still loops one backend call per batch element
//! via the Backend::matmul_batch trait default) vs. FastCpuBackend's
//! matmul_batch override (one thread::scope for the whole batch). Not a
//! correctness check (see ferro-core's op_bmm.rs for that); run with
//! --release to get meaningful numbers.

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

// Best-of-N wall time, matching ferro-fastcpu's bench.rs convention (this
// sandbox's CPU scheduling is noisy enough that a single sample is useless).
fn best_of(runs: usize, mut f: impl FnMut()) -> std::time::Duration {
    let mut best = std::time::Duration::MAX;
    for _ in 0..runs {
        let t0 = std::time::Instant::now();
        f();
        best = best.min(t0.elapsed());
    }
    best
}

#[test]
#[ignore]
fn bmm_perf_naive_vs_fastcpu() {
    let (batch, m, k, n) = (16usize, 128usize, 128usize, 128usize);
    let a = Tensor::from_vec(lcg_fill(1, batch * m * k), &[batch, m, k]).unwrap();
    let b = Tensor::from_vec(lcg_fill(2, batch * k * n), &[batch, k, n]).unwrap();

    let naive_dur = best_of(5, || {
        std::hint::black_box(a.bmm(&b).unwrap());
    });
    let naive_result = a.bmm(&b).unwrap();

    // Only the swappable single-matmul kernel installed: bmm still calls it
    // once per batch element via the Backend::matmul_batch trait default,
    // so this measures the per-call spawn/join overhead the batched path
    // below is meant to eliminate.
    ferro_fastcpu::install();
    let per_call_dur = best_of(5, || {
        std::hint::black_box(a.bmm(&b).unwrap());
    });
    let per_call_result = a.bmm(&b).unwrap();

    // FastCpuBackend registered: matmul_batch parallelizes the whole batch
    // under one thread::scope instead of one scope per batch element.
    ferro_fastcpu::install_backend();
    let batched_dur = best_of(5, || {
        std::hint::black_box(a.bmm(&b).unwrap());
    });
    let batched_result = a.bmm(&b).unwrap();

    let tol = 1e-2;
    let assert_close = |lhs: &Tensor, rhs: &Tensor, what: &str| {
        for (x, y) in lhs.to_vec().iter().zip(rhs.to_vec().iter()) {
            assert!((x - y).abs() <= tol * y.abs().max(1.0), "{what}: diverged {x} vs {y}");
        }
    };
    assert_close(&naive_result, &per_call_result, "per-call-matmul vs naive");
    assert_close(&naive_result, &batched_result, "matmul_batch vs naive");

    println!(
        "bmm forward batch={batch} m={m} k={k} n={n}: naive-cpu={naive_dur:?} fastcpu-per-call={per_call_dur:?} fastcpu-matmul_batch={batched_dur:?}"
    );
}
