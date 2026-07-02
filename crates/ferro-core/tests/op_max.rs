use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn max_value() {
    let a = Tensor::from_vec(vec![1.0, 5.0, 3.0], &[3]).unwrap();
    assert!((a.max().unwrap().item() - 5.0).abs() < 1e-6);
}

#[test]
fn max_nan_propagates() {
    let a = Tensor::from_vec(vec![1.0, f32::NAN, 0.5], &[3]).unwrap();
    assert!(a.max().unwrap().item().is_nan());
}

#[test]
fn max_all_nan() {
    let a = Tensor::from_vec(vec![f32::NAN, f32::NAN], &[2]).unwrap();
    assert!(a.max().unwrap().item().is_nan());
}

#[test]
fn max_empty_is_err() {
    let a = Tensor::from_vec(vec![], &[0]).unwrap();
    assert!(a.max().is_err());
}

#[test]
fn max_grad() {
    let a = Tensor::from_vec(vec![0.2, -1.0, 1.7, 0.5], &[4]).unwrap();
    grad_check(&[a], |t| t[0].max().unwrap());
}
