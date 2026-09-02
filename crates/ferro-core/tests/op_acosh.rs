use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn acosh_values() {
    let a = Tensor::from_vec(vec![1.5, 2.0, 3.0], &[3]).unwrap();
    let got = a.acosh().unwrap().to_vec();
    assert!((got[0] - 1.5f32.acosh()).abs() < 1e-5);
    assert!((got[1] - 2.0f32.acosh()).abs() < 1e-5);
    assert!((got[2] - 3.0f32.acosh()).abs() < 1e-5);
}

#[test]
fn acosh_grad() {
    let a = Tensor::from_vec(vec![1.5, 2.0, 2.5, 3.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].acosh().unwrap().sum());
}
