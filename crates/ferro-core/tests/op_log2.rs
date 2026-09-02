use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn log2_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 8.0], &[3]).unwrap();
    let got = a.log2().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 1.0).abs() < 1e-5);
    assert!((got[2] - 3.0).abs() < 1e-5);
}

#[test]
fn log2_grad() {
    let a = Tensor::from_vec(vec![0.5, 1.5, 2.0, 3.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].log2().unwrap().sum());
}
