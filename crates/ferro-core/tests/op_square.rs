use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn square_values() {
    let a = Tensor::from_vec(vec![2.0, -3.0, 0.5], &[3]).unwrap();
    let got = a.square().unwrap().to_vec();
    assert!((got[0] - 4.0).abs() < 1e-5);
    assert!((got[1] - 9.0).abs() < 1e-5);
    assert!((got[2] - 0.25).abs() < 1e-5);
}

#[test]
fn square_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.5, -2.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].square().unwrap().sum());
}
