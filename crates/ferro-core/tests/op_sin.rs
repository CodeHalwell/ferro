use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn sin_values() {
    let a = Tensor::from_vec(vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI], &[3]).unwrap();
    let got = a.sin().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 1.0).abs() < 1e-5);
    assert!((got[2] - 0.0).abs() < 1e-5);
}

#[test]
fn sin_grad() {
    let a = Tensor::from_vec(vec![0.3, 0.7, -0.5, 1.1], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].sin().unwrap().sum());
}
