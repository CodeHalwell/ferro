use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn log_sigmoid_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let got = a.log_sigmoid().unwrap().to_vec();
    let expect = |x: f32| (1.0 / (1.0 + (-x).exp())).ln();
    assert!((got[0] - expect(0.0)).abs() < 1e-5);
    assert!((got[1] - expect(1.0)).abs() < 1e-5);
    assert!((got[2] - expect(-1.0)).abs() < 1e-5);
}

#[test]
fn log_sigmoid_large_input_no_underflow() {
    let a = Tensor::from_vec(vec![-50.0, 50.0], &[2]).unwrap();
    let got = a.log_sigmoid().unwrap().to_vec();
    assert!(got[0].is_finite());
    assert!(got[1].is_finite());
    assert!((got[0] - (-50.0)).abs() < 1e-3);
    assert!((got[1] - 0.0).abs() < 1e-5);
}

#[test]
fn log_sigmoid_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, -1.5], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].log_sigmoid().unwrap().sum());
}
