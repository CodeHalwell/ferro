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
fn bce_with_logits_loss_grad_at_zero_logit() {
    // At x = 0, composing through relu/abs (both with zero subgradient at
    // their kink) would silently drop the sigmoid(0) = 0.5 term and give
    // dL/dx = -t instead of the correct sigmoid(0) - t = 0.5 - t.
    let x = Tensor::from_vec(vec![0.0, 0.0], &[2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let t = Tensor::from_vec(vec![0.3, 0.8], &[2]).unwrap();
    x.bce_with_logits_loss(&t).unwrap().backward();
    let g = x.grad().unwrap().to_vec();
    let n = 2.0f32;
    assert!((g[0] - (0.5 - 0.3) / n).abs() < 1e-5);
    assert!((g[1] - (0.5 - 0.8) / n).abs() < 1e-5);
}

#[test]
fn bce_with_logits_loss_large_logits_finite() {
    let x = Tensor::from_vec(vec![50.0, -50.0], &[2]).unwrap();
    let t = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let got = x.bce_with_logits_loss(&t).unwrap().item();
    assert!(got.is_finite());
}
