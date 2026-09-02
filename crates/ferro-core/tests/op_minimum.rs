use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn minimum_values() {
    let a = Tensor::from_vec(vec![1.0, 5.0, -2.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![3.0, 2.0, -2.0], &[3]).unwrap();
    let got = a.minimum(&b).unwrap().to_vec();
    assert!((got[0] - 1.0).abs() < 1e-5);
    assert!((got[1] - 2.0).abs() < 1e-5);
    assert!((got[2] - -2.0).abs() < 1e-5);
}

#[test]
fn minimum_values_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 5.0, -2.0, 4.0, 0.0, 3.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![3.0, 2.0, -2.0], &[3]).unwrap();
    let got = a.minimum(&b).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    assert_eq!(got.to_vec(), vec![1.0, 2.0, -2.0, 3.0, 0.0, -2.0]);
}

#[test]
fn minimum_grad() {
    let a = Tensor::from_vec(vec![0.5, 1.5, 2.0, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![2.0, 0.3, 3.5, 1.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].minimum(&t[1]).unwrap().sum());
}
