use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn hardshrink_values() {
    let a = Tensor::from_vec(vec![-1.0, 0.3, 2.0], &[3]).unwrap();
    let got = a.hardshrink(0.5).unwrap().to_vec();
    assert!((got[0] - (-1.0)).abs() < 1e-5);
    assert!((got[1] - 0.0).abs() < 1e-5);
    assert!((got[2] - 2.0).abs() < 1e-5);
}

#[test]
fn hardshrink_grad() {
    let a = Tensor::from_vec(vec![-2.0, -0.1, 0.1, 2.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].hardshrink(0.5).unwrap().sum());
}

#[test]
fn hardshrink_negative_lambd_errors() {
    let a = Tensor::from_vec(vec![1.0], &[1]).unwrap();
    assert!(a.hardshrink(-0.1).is_err());
}
