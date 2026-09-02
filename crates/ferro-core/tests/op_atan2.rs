use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn atan2_values() {
    let a = Tensor::from_vec(vec![1.0, 1.0, -1.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![1.0, -1.0, 1.0], &[3]).unwrap();
    let got = a.atan2(&b).unwrap().to_vec();
    assert!((got[0] - 1.0f32.atan2(1.0)).abs() < 1e-5);
    assert!((got[1] - 1.0f32.atan2(-1.0)).abs() < 1e-5);
    assert!((got[2] - (-1.0f32).atan2(1.0)).abs() < 1e-5);
}

#[test]
fn atan2_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, -1.5, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![2.0, -1.0, 1.0, -2.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].atan2(&t[1]).unwrap().sum());
}
