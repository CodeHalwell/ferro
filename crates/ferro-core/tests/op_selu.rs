use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

const ALPHA: f32 = 1.6732632423543772848170429916717;
const SCALE: f32 = 1.0507009873554804934193349852946;

#[test]
fn selu_values() {
    let a = Tensor::from_vec(vec![-2.0, 0.5, 3.0], &[3]).unwrap();
    let got = a.selu().unwrap().to_vec();
    assert!((got[0] - SCALE * ALPHA * (-2.0f32).exp_m1()).abs() < 1e-5);
    assert!((got[1] - SCALE * 0.5).abs() < 1e-5);
    assert!((got[2] - SCALE * 3.0).abs() < 1e-5);
}

#[test]
fn selu_grad() {
    let a = Tensor::from_vec(vec![-1.2, -0.4, 0.6, 1.8], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].selu().unwrap().sum());
}
