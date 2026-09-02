use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn erf_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let got = a.erf().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 0.8427007929497149).abs() < 1e-5);
    assert!((got[2] - (-0.8427007929497149)).abs() < 1e-5);
}

#[test]
fn erf_grad() {
    let a = Tensor::from_vec(vec![0.4, -0.8, 1.3, -1.7], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].erf().unwrap().sum());
}
