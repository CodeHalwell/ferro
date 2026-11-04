use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn log_softmax_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]).unwrap();
    let y = a.log_softmax(1).unwrap().to_vec();

    // exp(log_softmax) rows sum to 1.
    for r in 0..2 {
        let s: f32 = (0..3).map(|c| y[r * 3 + c].exp()).sum();
        assert!((s - 1.0).abs() < 1e-5, "row {r} sum {s}");
    }

    // Equals log(softmax) computed independently.
    let x = a.to_vec();
    for r in 0..2 {
        let row = &x[r * 3..r * 3 + 3];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = row.iter().map(|v| (v - m).exp()).sum();
        for c in 0..3 {
            let expected = (row[c] - m) - denom.ln();
            assert!((y[r * 3 + c] - expected).abs() < 1e-5);
        }
    }
}

#[test]
fn log_softmax_grad() {
    let a = Tensor::from_vec(vec![0.3, -1.2, 0.7, 2.0, -0.5, 1.1], &[2, 3]).unwrap();

    // Plain sum: exercises the -softmax * sum_g term (sum_g != 0).
    grad_check(&[a.clone()], |t| t[0].log_softmax(1).unwrap().sum());

    // Weighted by input.
    grad_check(&[a.clone()], |t| {
        t[0].log_softmax(1).unwrap().mul(&t[0]).unwrap().sum()
    });

    // Along dim 0.
    grad_check(&[a], |t| t[0].log_softmax(0).unwrap().sum());
}

#[test]
fn large_common_offsets_preserve_normalization_and_gradients() {
    for offset in [1e8, -1e8, 3e38] {
        for dim in 0..3 {
            let x = Tensor::full(&[2,3,2], offset).requires_grad_(true).unwrap();
            let y = x.log_softmax(dim).unwrap();
            let width = x.shape()[dim];
            for value in y.to_vec() {
                assert!((value + (width as f32).ln()).abs() < 2e-6, "offset={offset}, dim={dim}, value={value}");
            }
            let seed = Tensor::from_vec((0..12).map(|i| (i % 5) as f32 - 2.).collect(), &[2,3,2]).unwrap();
            y.backward_with(&seed);
            let expected = seed.sub(&seed.sum_dim(dim, true).unwrap().div(&Tensor::scalar(width as f32)).unwrap()).unwrap();
            for (actual, expected) in x.grad().unwrap().to_vec().iter().zip(expected.to_vec()) {
                assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
            }
        }
    }
}
