use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn leaky_relu_values() {
    let a = Tensor::from_vec(vec![-2.0, 0.0, 3.0], &[3]).unwrap();
    let got = a.leaky_relu(0.1).unwrap().to_vec();
    assert!((got[0] - (-0.2)).abs() < 1e-5);
    assert!((got[1] - 0.0).abs() < 1e-5);
    assert!((got[2] - 3.0).abs() < 1e-5);
}

#[test]
fn leaky_relu_grad() {
    let a = Tensor::from_vec(vec![1.3, -0.7, 2.4, -1.5], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].leaky_relu(0.1).unwrap().sum());
}
