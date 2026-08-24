use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn bmm_values() {
    // batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[10,11]]
    // batch 1: [[1,1,1],[2,2,2]] @ [[1,2],[3,4],[5,6]] = [[9,12],[18,24]]
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        &[2, 2, 3],
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3, 2],
    )
    .unwrap();
    let c = a.bmm(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2, 2]);
    assert_eq!(
        c.to_vec(),
        vec![4.0, 5.0, 10.0, 11.0, 9.0, 12.0, 18.0, 24.0]
    );
}

#[test]
fn bmm_shape_error() {
    let a = Tensor::from_vec(vec![0.0; 2 * 2 * 3], &[2, 2, 3]).unwrap();
    // inner dim mismatch: (2,2,3) @ (2,4,2)
    let b = Tensor::from_vec(vec![0.0; 2 * 4 * 2], &[2, 4, 2]).unwrap();
    assert!(a.bmm(&b).is_err());
    // batch mismatch: (2,2,3) @ (3,3,2)
    let b2 = Tensor::from_vec(vec![0.0; 3 * 3 * 2], &[3, 3, 2]).unwrap();
    assert!(a.bmm(&b2).is_err());
}

#[test]
fn bmm_grad() {
    let a = Tensor::from_vec(
        vec![
            0.5, -1.0, 2.0, 1.5, 0.3, -0.7, 1.0, 2.0, -1.0, 0.5, 0.8, -0.2,
        ],
        &[2, 2, 3],
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![
            1.0, -0.5, 0.2, 1.3, -1.0, 0.7, 0.4, 0.9, -0.3, 1.1, 0.6, -0.8,
        ],
        &[2, 3, 2],
    )
    .unwrap();
    grad_check(&[a, b], |t| t[0].bmm(&t[1]).unwrap().sum());
}

// Deterministic fill, no external rand dep (mirrors ferro-fastcpu's test helper).
fn lcg_fill(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (&x, &y)) in actual.iter().zip(expected).enumerate() {
        let tol = 1e-4 * y.abs().max(1.0);
        assert!(
            (x - y).abs() <= tol,
            "{what}: mismatch at {i}: {x} vs {y} (tol {tol})"
        );
    }
}

// Pre-dispatch reference forward (the old bmm.rs ijp triple loop).
fn naive_batched_matmul(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
        for i in 0..m {
            for p in 0..k {
                let av = a[ao + i * k + p];
                for j in 0..n {
                    c[co + i * n + j] += av * b[bo + p * n + j];
                }
            }
        }
    }
    c
}

// Pre-dispatch reference backward (the old bmm.rs dA/dB triple loops).
fn naive_batched_backward(
    a: &[f32],
    b: &[f32],
    g: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut da = vec![0.0f32; batch * m * k];
    for bi in 0..batch {
        let (go, bo, ao) = (bi * m * n, bi * k * n, bi * m * k);
        for i in 0..m {
            for p in 0..k {
                let mut s = 0.0f32;
                for j in 0..n {
                    s += g[go + i * n + j] * b[bo + p * n + j];
                }
                da[ao + i * k + p] = s;
            }
        }
    }

    let mut db = vec![0.0f32; batch * k * n];
    for bi in 0..batch {
        let (go, ao, bo) = (bi * m * n, bi * m * k, bi * k * n);
        for p in 0..k {
            for j in 0..n {
                let mut s = 0.0f32;
                for i in 0..m {
                    s += a[ao + i * k + p] * g[go + i * n + j];
                }
                db[bo + p * n + j] = s;
            }
        }
    }
    (da, db)
}

#[test]
fn bmm_dispatch_parity() {
    let dims = [
        (1, 1, 1),
        (5, 5, 5),
        (17, 17, 17),
        (64, 64, 64),
        (1, 5, 17),
        (5, 1, 17),
        (5, 17, 1),
        (1, 64, 64),
        (64, 1, 64),
        (64, 64, 1),
        (17, 64, 5),
        (64, 5, 17),
        (5, 64, 17),
        (1, 1, 64),
        (1, 64, 1),
        (64, 1, 1),
        (1, 17, 5),
    ];
    for batch in [1usize, 7usize] {
        for (i, &(m, k, n)) in dims.iter().enumerate() {
            let what = format!("batch={batch} m={m} k={k} n={n}");
            let a_data = lcg_fill(1000 + i as u64, batch * m * k);
            let b_data = lcg_fill(2000 + i as u64, batch * k * n);
            let a = Tensor::from_vec(a_data.clone(), &[batch, m, k])
                .unwrap()
                .requires_grad_(true)
                .unwrap();
            let b = Tensor::from_vec(b_data.clone(), &[batch, k, n])
                .unwrap()
                .requires_grad_(true)
                .unwrap();

            let c = a.bmm(&b).unwrap();
            let expected_c = naive_batched_matmul(&a_data, &b_data, batch, m, k, n);
            assert_close(&c.to_vec(), &expected_c, &format!("{what} forward"));

            c.sum().backward();
            let g_data = vec![1.0f32; batch * m * n]; // d(sum)/dC is all-ones.
            let (expected_da, expected_db) =
                naive_batched_backward(&a_data, &b_data, &g_data, batch, m, k, n);
            assert_close(
                &a.grad().unwrap().to_vec(),
                &expected_da,
                &format!("{what} dA"),
            );
            assert_close(
                &b.grad().unwrap().to_vec(),
                &expected_db,
                &format!("{what} dB"),
            );
        }
    }
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
fn bmm_perf_old_vs_dispatch() {
    let (batch, m, k, n) = (16usize, 128usize, 128usize, 128usize);
    let a_data = lcg_fill(1, batch * m * k);
    let b_data = lcg_fill(2, batch * k * n);
    let a = Tensor::from_vec(a_data.clone(), &[batch, m, k]).unwrap();
    let b = Tensor::from_vec(b_data.clone(), &[batch, k, n]).unwrap();

    let old_dur = best_of(5, || {
        std::hint::black_box(naive_batched_matmul(&a_data, &b_data, batch, m, k, n));
    });
    let new_dur = best_of(5, || {
        std::hint::black_box(a.bmm(&b).unwrap());
    });

    let old_c = naive_batched_matmul(&a_data, &b_data, batch, m, k, n);
    assert_close(&a.bmm(&b).unwrap().to_vec(), &old_c, "perf sanity check");
    println!(
        "bmm forward batch={batch} m={m} k={k} n={n}: old-naive={old_dur:?} new-dispatch={new_dur:?}"
    );
}
