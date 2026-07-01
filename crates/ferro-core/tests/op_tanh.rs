use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn tanh_values() {
    let a = Tensor::from_vec(vec![0.0, 0.5, -1.0, 2.0], &[4]).unwrap();
    let got = a.tanh().to_vec();
    let expected: Vec<f32> = vec![0.0, 0.5, -1.0, 2.0]
        .into_iter()
        .map(f32::tanh)
        .collect();
    for (g, e) in got.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
    }
}

#[test]
fn tanh_grad() {
    let a = Tensor::from_vec(vec![-1.0, -0.3, 0.4, 1.2], &[4]).unwrap();
    grad_check(&[a], |t| t[0].tanh().sum());
}
