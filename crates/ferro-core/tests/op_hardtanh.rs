use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn hardtanh_values() {
    let a = Tensor::from_vec(vec![-3.0, -0.5, 0.5, 3.0], &[4]).unwrap();
    let got = a.hardtanh(-1.0, 1.0).unwrap().to_vec();
    assert!((got[0] - (-1.0)).abs() < 1e-5);
    assert!((got[1] - (-0.5)).abs() < 1e-5);
    assert!((got[2] - 0.5).abs() < 1e-5);
    assert!((got[3] - 1.0).abs() < 1e-5);
}

#[test]
fn hardtanh_grad() {
    let a = Tensor::from_vec(vec![-2.0, -0.3, 0.6, 2.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].hardtanh(-1.0, 1.0).unwrap().sum());
}
