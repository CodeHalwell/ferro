use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn trunc_values() {
    let a = Tensor::from_vec(vec![2.7, -2.7, 0.4], &[3]).unwrap();
    let got = a.trunc().unwrap().to_vec();
    assert!((got[0] - 2.0).abs() < 1e-5);
    assert!((got[1] - -2.0).abs() < 1e-5);
    assert!((got[2] - 0.0).abs() < 1e-5);
}

#[test]
fn trunc_grad() {
    let a = Tensor::from_vec(vec![0.3, 1.7, -0.5, 2.4], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].trunc().unwrap().sum());
}
