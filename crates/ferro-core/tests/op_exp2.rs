use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn exp2_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, 3.0], &[3]).unwrap();
    let got = a.exp2().unwrap().to_vec();
    assert!((got[0] - 1.0).abs() < 1e-5);
    assert!((got[1] - 2.0).abs() < 1e-5);
    assert!((got[2] - 8.0).abs() < 1e-5);
}

#[test]
fn exp2_grad() {
    let a = Tensor::from_vec(vec![0.3, 0.7, -0.5, 1.1], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].exp2().unwrap().sum());
}
