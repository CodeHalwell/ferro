use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn softplus_values() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let got = a.softplus().unwrap().to_vec();
    assert!((got[0] - 2.0f32.ln()).abs() < 1e-5);
    assert!((got[1] - (1.0 + 1.0f32.exp()).ln()).abs() < 1e-5);
    assert!((got[2] - (1.0 + (-1.0f32).exp()).ln()).abs() < 1e-5);
}

#[test]
fn softplus_large_input_no_overflow() {
    let a = Tensor::from_vec(vec![100.0, 1000.0, 50.0], &[3]).unwrap();
    let got = a.softplus().unwrap().to_vec();
    assert!(got[0].is_finite());
    assert!(got[1].is_finite());
    assert!(got[2].is_finite());
    assert!((got[0] - 100.0).abs() < 1e-5);
    assert!((got[1] - 1000.0).abs() < 1e-5);
    assert!((got[2] - 50.0).abs() < 1e-5);
}

#[test]
fn softplus_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.2, -1.5], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].softplus().unwrap().sum());
}
