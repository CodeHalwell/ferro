use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn sign_values() {
    let a = Tensor::from_vec(vec![-3.0, 0.0, 2.5], &[3]).unwrap();
    let got = a.sign().unwrap().to_vec();
    assert!((got[0] - (-1.0)).abs() < 1e-5);
    assert!((got[1] - 0.0).abs() < 1e-5);
    assert!((got[2] - 1.0).abs() < 1e-5);
}

#[test]
fn sign_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.5, -2.4], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].sign().unwrap().sum());
}
