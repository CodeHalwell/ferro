use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn celu_values() {
    let a = Tensor::from_vec(vec![-2.0, 0.0, 3.0], &[3]).unwrap();
    let got = a.celu(1.5).unwrap().to_vec();
    assert!((got[0] - 1.5 * (-2.0f32 / 1.5).exp_m1()).abs() < 1e-5);
    assert!((got[1] - 0.0).abs() < 1e-5);
    assert!((got[2] - 3.0).abs() < 1e-5);
}

#[test]
fn celu_grad() {
    let a = Tensor::from_vec(vec![-1.2, -0.4, 0.5, 1.3], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].celu(1.5).unwrap().sum());
}

#[test]
fn celu_zero_alpha_errors() {
    let a = Tensor::from_vec(vec![1.0], &[1]).unwrap();
    assert!(a.celu(0.0).is_err());
}
