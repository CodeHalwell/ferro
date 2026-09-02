use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn log10_values() {
    let a = Tensor::from_vec(vec![1.0, 10.0, 100.0], &[3]).unwrap();
    let got = a.log10().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 1.0).abs() < 1e-5);
    assert!((got[2] - 2.0).abs() < 1e-5);
}

#[test]
fn log10_grad() {
    let a = Tensor::from_vec(vec![0.5, 1.5, 2.0, 3.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].log10().unwrap().sum());
}
