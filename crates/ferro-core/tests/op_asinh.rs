use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn asinh_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
    let got = a.asinh().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 0.88137359).abs() < 1e-5);
    assert!((got[2] - 1.44363548).abs() < 1e-5);
}

#[test]
fn asinh_grad() {
    let a = Tensor::from_vec(vec![0.3, 0.7, -0.5, 1.1], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].asinh().unwrap().sum());
}
