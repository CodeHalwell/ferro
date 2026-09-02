use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn ceil_values() {
    let a = Tensor::from_vec(vec![1.2, -1.2, 2.0], &[3]).unwrap();
    let got = a.ceil().unwrap().to_vec();
    assert!((got[0] - 2.0).abs() < 1e-5);
    assert!((got[1] - (-1.0)).abs() < 1e-5);
    assert!((got[2] - 2.0).abs() < 1e-5);
}

#[test]
fn ceil_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 2.4, -1.6], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].ceil().unwrap().sum());
}
