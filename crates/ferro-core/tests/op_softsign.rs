use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn softsign_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -2.0], &[3]).unwrap();
    let got = a.softsign().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 0.5).abs() < 1e-5);
    assert!((got[2] - (-2.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn softsign_grad() {
    let a = Tensor::from_vec(vec![-1.7, -0.6, 0.9, 2.3], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].softsign().unwrap().sum());
}
