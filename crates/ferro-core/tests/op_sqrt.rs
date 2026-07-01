use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn sqrt_values() {
    let a = Tensor::from_vec(vec![4.0, 9.0, 16.0], &[3]).unwrap();
    let got = a.sqrt().to_vec();
    assert!((got[0] - 2.0).abs() < 1e-5);
    assert!((got[1] - 3.0).abs() < 1e-5);
    assert!((got[2] - 4.0).abs() < 1e-5);
}

#[test]
fn sqrt_grad() {
    let a = Tensor::from_vec(vec![0.5, 1.0, 2.0, 4.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].sqrt().sum());
}
