use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn sinc_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, 0.5], &[3]).unwrap();
    let got = a.sinc().unwrap().to_vec();
    assert!((got[0] - 1.0).abs() < 1e-5);
    assert!((got[1] - 0.0).abs() < 1e-5);
    assert!((got[2] - std::f32::consts::FRAC_2_PI).abs() < 1e-5);
}

#[test]
fn sinc_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.6, 1.2, -1.7], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].sinc().unwrap().sum());
}
