use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn hardsigmoid_values() {
    let a = Tensor::from_vec(vec![-4.0, 0.0, 1.5, 4.0], &[4]).unwrap();
    let got = a.hardsigmoid().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 0.5).abs() < 1e-5);
    assert!((got[2] - 0.75).abs() < 1e-5);
    assert!((got[3] - 1.0).abs() < 1e-5);
}

#[test]
fn hardsigmoid_grad() {
    let a = Tensor::from_vec(vec![-1.0, 0.5, 1.5, 2.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].hardsigmoid().unwrap().sum());
}
