use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn hardswish_values() {
    let a = Tensor::from_vec(vec![-5.0, 0.0, 1.5, 4.0], &[4]).unwrap();
    let got = a.hardswish().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 0.0).abs() < 1e-5);
    assert!((got[2] - 1.125).abs() < 1e-5);
    assert!((got[3] - 4.0).abs() < 1e-5);
}

#[test]
fn hardswish_grad() {
    let a = Tensor::from_vec(vec![-5.0, -1.0, 0.5, 4.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].hardswish().unwrap().sum());
}
