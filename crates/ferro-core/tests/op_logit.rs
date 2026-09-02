use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn logit_values() {
    let a = Tensor::from_vec(vec![0.5, 0.25, 0.75], &[3]).unwrap();
    let got = a.logit().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - (0.25f32 / 0.75).ln()).abs() < 1e-5);
    assert!((got[2] - (0.75f32 / 0.25).ln()).abs() < 1e-5);
}

#[test]
fn logit_grad() {
    let a = Tensor::from_vec(vec![0.2, 0.8, 0.35, 0.65], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].logit().unwrap().sum());
}
