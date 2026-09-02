use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn asin_values() {
    let a = Tensor::from_vec(vec![0.0, 0.5, -0.5], &[3]).unwrap();
    let got = a.asin().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - std::f32::consts::FRAC_PI_6).abs() < 1e-5);
    assert!((got[2] + std::f32::consts::FRAC_PI_6).abs() < 1e-5);
}

#[test]
fn asin_grad() {
    let a = Tensor::from_vec(vec![-0.6, -0.2, 0.2, 0.6], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].asin().unwrap().sum());
}
