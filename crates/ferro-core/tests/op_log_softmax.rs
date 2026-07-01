use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn log_softmax_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]).unwrap();
    let y = a.log_softmax(1).to_vec();

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
    grad_check(&[a.clone()], |t| t[0].log_softmax(1).sum());

    // Weighted by input.
    grad_check(&[a.clone()], |t| {
        t[0].log_softmax(1).mul(&t[0]).unwrap().sum()
    });

    // Along dim 0.
    grad_check(&[a], |t| t[0].log_softmax(0).sum());
}
