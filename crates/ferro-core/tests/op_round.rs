use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn round_values() {
    let a = Tensor::from_vec(vec![0.5, 1.5, 2.5, -0.5], &[4]).unwrap();
    let got = a.round().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 2.0).abs() < 1e-5);
    assert!((got[2] - 2.0).abs() < 1e-5);
    assert!((got[3] - 0.0).abs() < 1e-5);
}

#[test]
fn round_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, -1.8], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].round().unwrap().sum());
}
