use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn expm1_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let got = a.expm1().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 1.0f32.exp_m1()).abs() < 1e-5);
    assert!((got[2] - (-1.0f32).exp_m1()).abs() < 1e-5);
}

#[test]
fn expm1_grad() {
    let a = Tensor::from_vec(vec![0.3, 0.7, -0.5, 1.1], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].expm1().unwrap().sum());
}
