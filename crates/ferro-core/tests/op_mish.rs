use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn mish_values() {
    let a = Tensor::from_vec(vec![-2.0, 0.0, 1.0, 25.0], &[4]).unwrap();
    let got = a.mish().unwrap().to_vec();
    let expect = |x: f32| x * (1.0 + x.exp()).ln().tanh();
    assert!((got[0] - expect(-2.0)).abs() < 1e-5);
    assert!((got[1] - expect(0.0)).abs() < 1e-5);
    assert!((got[2] - expect(1.0)).abs() < 1e-5);
    assert!((got[3] - 25.0 * 25.0f32.tanh()).abs() < 1e-5);
}

#[test]
fn mish_grad() {
    let a = Tensor::from_vec(vec![-1.5, -0.3, 0.7, 2.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].mish().unwrap().sum());
}
