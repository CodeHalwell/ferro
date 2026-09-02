use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn acos_values() {
    let a = Tensor::from_vec(vec![0.0, 0.5, -0.5], &[3]).unwrap();
    let got = a.acos().unwrap().to_vec();
    assert!((got[0] - 0.0f32.acos()).abs() < 1e-5);
    assert!((got[1] - 0.5f32.acos()).abs() < 1e-5);
    assert!((got[2] - (-0.5f32).acos()).abs() < 1e-5);
}

#[test]
fn acos_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.3, 0.6, -0.6], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].acos().unwrap().sum());
}
