use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn tanhshrink_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -2.0], &[3]).unwrap();
    let got = a.tanhshrink().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - (1.0 - 1.0f32.tanh())).abs() < 1e-5);
    assert!((got[2] - (-2.0 - (-2.0f32).tanh())).abs() < 1e-5);
}

#[test]
fn tanhshrink_grad() {
    let a = Tensor::from_vec(vec![1.0, -1.5, 2.0, -0.8], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].tanhshrink().unwrap().sum());
}
