use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn maximum_values() {
    let a = Tensor::from_vec(vec![1.0, 5.0, -2.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 2.0, -3.0], &[3]).unwrap();
    let got = a.maximum(&b).unwrap().to_vec();
    assert!((got[0] - 4.0).abs() < 1e-5);
    assert!((got[1] - 5.0).abs() < 1e-5);
    assert!((got[2] - (-2.0)).abs() < 1e-5);
}

#[test]
fn maximum_grad() {
    let a = Tensor::from_vec(vec![1.0, 5.0, 3.0, -1.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 2.0, -3.0, 2.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].maximum(&t[1]).unwrap().sum());
}

#[test]
fn maximum_grad_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 5.0, 3.0, 7.0, -2.0, 9.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, -6.0, 3.5], &[3]).unwrap();
    grad_check(&[a, b], |t| t[0].maximum(&t[1]).unwrap().sum());
}
