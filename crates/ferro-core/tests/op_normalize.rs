use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn normalize_values() {
    let a = Tensor::from_vec(vec![3.0, 4.0, 0.0, 0.0, 0.0, 2.0], &[2, 3]).unwrap();
    let got = a.normalize(1, 1e-8).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    let v = got.to_vec();
    let norm0 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let norm1 = (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]).sqrt();
    assert!((norm0 - 1.0).abs() < 1e-5);
    assert!((norm1 - 1.0).abs() < 1e-5);
    assert!((v[0] - 0.6).abs() < 1e-5);
    assert!((v[1] - 0.8).abs() < 1e-5);
    assert!((v[5] - 1.0).abs() < 1e-5);
}

#[test]
fn normalize_out_of_range_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(a.normalize(1, 1e-8).is_err());
}

#[test]
fn normalize_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, -0.5, 0.7], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].normalize(1, 1e-8).unwrap().sum());
}
