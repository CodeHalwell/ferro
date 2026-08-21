use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn gelu_values() {
    let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
    let got = a.gelu().to_vec();
    // Reference: torch.nn.functional.gelu(x, approximate="tanh").
    let want = [-0.15881, 0.0, 0.84119, 1.95460];
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
    }
}

#[test]
fn gelu_grad() {
    let a = Tensor::from_vec(vec![-1.5, -0.4, 0.3, 0.9, 1.7, -2.1], &[2, 3]).unwrap();
    grad_check(&[a], |t| t[0].gelu().sum());
}
