use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn abs_values() {
    let a = Tensor::from_vec(vec![-3.0, 2.0], &[2]).unwrap();
    let got = a.abs().to_vec();
    assert!((got[0] - 3.0).abs() < 1e-5);
    assert!((got[1] - 2.0).abs() < 1e-5);
}

#[test]
fn abs_grad() {
    let a = Tensor::from_vec(vec![-2.0, -0.5, 0.7, 1.5], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].abs().sum());
}
