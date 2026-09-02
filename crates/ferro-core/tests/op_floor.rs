use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn floor_values() {
    let a = Tensor::from_vec(vec![0.3, 1.4, -0.6, 2.4], &[4]).unwrap();
    let got = a.floor().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 1.0).abs() < 1e-5);
    assert!((got[2] - (-1.0)).abs() < 1e-5);
    assert!((got[3] - 2.0).abs() < 1e-5);
}

#[test]
fn floor_grad() {
    let a = Tensor::from_vec(vec![0.3, 1.4, -0.6, 2.4], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].floor().unwrap().sum());
}
