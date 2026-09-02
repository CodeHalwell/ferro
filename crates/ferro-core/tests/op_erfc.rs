use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn erfc_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let got = a.erfc().unwrap().to_vec();
    assert!((got[0] - 1.0).abs() < 1e-5);
    assert!((got[1] - 0.15729920705028513).abs() < 1e-5);
    assert!((got[2] - 1.8427007929497148).abs() < 1e-5);
}

#[test]
fn erfc_grad() {
    let a = Tensor::from_vec(vec![0.3, 0.7, -0.5, 1.1], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].erfc().unwrap().sum());
}
