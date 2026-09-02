use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn atanh_values() {
    let a = Tensor::from_vec(vec![0.0, 0.5, -0.3], &[3]).unwrap();
    let got = a.atanh().unwrap().to_vec();
    assert!((got[0] - 0.0f32.atanh()).abs() < 1e-5);
    assert!((got[1] - 0.5f32.atanh()).abs() < 1e-5);
    assert!((got[2] - (-0.3f32).atanh()).abs() < 1e-5);
}

#[test]
fn atanh_grad() {
    let a = Tensor::from_vec(vec![0.2, -0.4, 0.6, -0.1], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].atanh().unwrap().sum());
}
