use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn elu_values() {
    let a = Tensor::from_vec(vec![-1.5, 0.0, 2.0], &[3]).unwrap();
    let got = a.elu(2.0).unwrap().to_vec();
    assert!((got[0] - 2.0 * (-1.5f32).exp_m1()).abs() < 1e-5);
    assert!((got[1] - 0.0).abs() < 1e-5);
    assert!((got[2] - 2.0).abs() < 1e-5);
}

#[test]
fn elu_grad() {
    let a = Tensor::from_vec(vec![-1.2, -0.4, 0.5, 1.3], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].elu(1.5).unwrap().sum());
}
