use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn heaviside_values() {
    let a = Tensor::from_vec(vec![-2.0, 0.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 7.0, 9.0], &[3]).unwrap();
    let got = a.heaviside(&b).unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 7.0).abs() < 1e-5);
    assert!((got[2] - 1.0).abs() < 1e-5);
}

#[test]
fn heaviside_values_broadcast() {
    let a = Tensor::from_vec(vec![-1.0, 0.0, 2.0, 0.0, -3.0, 4.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let got = a.heaviside(&b).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    assert_eq!(got.to_vec(), vec![0.0, 20.0, 1.0, 10.0, 0.0, 1.0]);
}

#[test]
fn heaviside_grad() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 3.0, -0.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.5, -1.0, 2.0, 3.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].heaviside(&t[1]).unwrap().sum());
}
