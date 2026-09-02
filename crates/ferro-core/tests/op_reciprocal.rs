use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn reciprocal_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, -4.0], &[3]).unwrap();
    let got = a.reciprocal().unwrap().to_vec();
    assert!((got[0] - 1.0).abs() < 1e-5);
    assert!((got[1] - 0.5).abs() < 1e-5);
    assert!((got[2] - (-0.25)).abs() < 1e-5);
}

#[test]
fn reciprocal_grad() {
    let a = Tensor::from_vec(vec![1.0, -1.5, 2.0, -3.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].reciprocal().unwrap().sum());
}
