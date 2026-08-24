use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn silu_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let y = a.silu().to_vec();
    assert!(y[0].abs() < 1e-6);
    // sigmoid(-1) = 1 / (1 + e) since exp(-1.0) is the small value; sanity-check
    // the constant: silu(-1) = -1 * sigmoid(-1).
    let sig = 1.0 / (1.0 + f32::exp(1.0)); // sigmoid(-1)
    assert!((y[2] + sig).abs() < 1e-6);
}

#[test]
fn silu_no_grad_path() {
    let a = Tensor::from_vec(vec![0.3, -0.7], &[2]).unwrap();
    assert!(!a.silu().requires_grad());
}

// Weighted-sum loss so every element gets a distinct gradient.
fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.2 + 0.31 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn silu_grad() {
    // Away from the zero kink in O(1)-magnitude territory.
    let a = Tensor::from_vec(vec![0.8, -1.3, 2.1, -0.4], &[2, 2]).unwrap();
    grad_check(&[a.clone()], |t| weighted_loss(t[0].clone().silu()));
}
