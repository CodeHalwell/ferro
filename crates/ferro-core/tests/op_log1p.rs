use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn log1p_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, std::f32::consts::E - 1.0], &[3]).unwrap();
    let got = a.log1p().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 2.0f32.ln()).abs() < 1e-5);
    assert!((got[2] - 1.0).abs() < 1e-5);
}

#[test]
fn log1p_grad() {
    let a = Tensor::from_vec(vec![0.5, 1.5, -0.5, 2.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].log1p().unwrap().sum());
}
