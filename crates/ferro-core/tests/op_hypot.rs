use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn hypot_values() {
    let a = Tensor::from_vec(vec![3.0, 5.0, 1.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 12.0, 1.0], &[3]).unwrap();
    let got = a.hypot(&b).unwrap().to_vec();
    assert!((got[0] - 5.0).abs() < 1e-5);
    assert!((got[1] - 13.0).abs() < 1e-5);
    assert!((got[2] - 2.0f32.sqrt()).abs() < 1e-5);
}

#[test]
fn hypot_grad() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![2.0, 1.0, -1.0, 1.5], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].hypot(&t[1]).unwrap().sum());
}
