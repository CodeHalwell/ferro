use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn bce_with_logits_loss_values() {
    let x = Tensor::from_vec(vec![0.0, 2.0, -3.0], &[3]).unwrap();
    let t = Tensor::from_vec(vec![0.5, 1.0, 0.0], &[3]).unwrap();
    let got = x.bce_with_logits_loss(&t).unwrap().item();
    assert!((got - 0.2895542).abs() < 1e-5);
}

#[test]
fn bce_with_logits_loss_grad() {
    let logits = Tensor::from_vec(vec![1.0, -0.5, 0.3, -2.0], &[2, 2]).unwrap();
    let target = Tensor::from_vec(vec![0.3, 0.7, 0.5, 0.2], &[2, 2]).unwrap();
    grad_check(&[logits, target], |t| t[0].bce_with_logits_loss(&t[1]).unwrap());
}

#[test]
fn bce_with_logits_loss_large_logits_finite() {
    let x = Tensor::from_vec(vec![50.0, -50.0], &[2]).unwrap();
    let t = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let got = x.bce_with_logits_loss(&t).unwrap().item();
    assert!(got.is_finite());
}
