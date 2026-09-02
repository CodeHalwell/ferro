use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn bce_loss_values() {
    let p = Tensor::from_vec(vec![0.5, 0.25, 0.75], &[3]).unwrap();
    let t = Tensor::from_vec(vec![1.0, 0.0, 0.5], &[3]).unwrap();
    let got = p.bce_loss(&t).unwrap().to_vec();
    let e0 = -(1.0f32 * 0.5f32.ln() + 0.0f32 * 0.5f32.ln());
    let e1 = -(0.0f32 * 0.25f32.ln() + 1.0f32 * 0.75f32.ln());
    let e2 = -(0.5f32 * 0.75f32.ln() + 0.5f32 * 0.25f32.ln());
    let expected = (e0 + e1 + e2) / 3.0;
    assert!((got[0] - expected).abs() < 1e-5);
}

#[test]
fn bce_loss_grad() {
    let p = Tensor::from_vec(vec![0.3, 0.5, 0.7, 0.6], &[2, 2]).unwrap();
    let t = Tensor::from_vec(vec![0.2, 0.8, 0.4, 0.6], &[2, 2]).unwrap();
    grad_check(&[p, t], |ts| ts[0].bce_loss(&ts[1]).unwrap());
}
