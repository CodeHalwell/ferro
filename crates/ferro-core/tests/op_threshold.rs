use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn threshold_values() {
    let a = Tensor::from_vec(vec![-2.0, 0.2, 3.0], &[3]).unwrap();
    let got = a.threshold(0.5, -1.0).unwrap().to_vec();
    assert!((got[0] - (-1.0)).abs() < 1e-5);
    assert!((got[1] - (-1.0)).abs() < 1e-5);
    assert!((got[2] - 3.0).abs() < 1e-5);
}

#[test]
fn threshold_grad() {
    let a = Tensor::from_vec(vec![-2.0, 0.2, 0.6, 3.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].threshold(0.5, -1.0).unwrap().sum());
}
