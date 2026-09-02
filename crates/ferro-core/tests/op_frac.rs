use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn frac_values() {
    let a = Tensor::from_vec(vec![1.75, -1.25, 3.0], &[3]).unwrap();
    let got = a.frac().unwrap().to_vec();
    assert!((got[0] - 0.75).abs() < 1e-5);
    assert!((got[1] - -0.25).abs() < 1e-5);
    assert!((got[2] - 0.0).abs() < 1e-5);
}

#[test]
fn frac_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, -1.8], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].frac().unwrap().sum());
}
