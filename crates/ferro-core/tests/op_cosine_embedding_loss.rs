use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn cosine_embedding_loss_values() {
    let x1 = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0], &[4, 2]).unwrap();
    let x2 = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[4, 2]).unwrap();
    let target = Tensor::from_vec(vec![1.0, 1.0, -1.0, -1.0], &[4]).unwrap();
    let got = x1.cosine_embedding_loss(&x2, &target, 0.5, 1e-8).unwrap();
    assert_eq!(got.shape(), &[] as &[usize]);
    // rows: 1 - 1 = 0; 1 - 0 = 1; max(0, 1/sqrt2 - 0.5); max(0, 0 - 0.5) = 0
    let want = (1.0 + (std::f32::consts::FRAC_1_SQRT_2 - 0.5)) / 4.0;
    assert!((got.item() - want).abs() < 1e-5);
}

#[test]
fn cosine_embedding_loss_rejects_scalar() {
    let s = Tensor::scalar(1.0);
    assert!(s.cosine_embedding_loss(&s, &s, 0.0, 1e-8).is_err());
}

#[test]
fn cosine_embedding_loss_grad() {
    // Row 0 is a positive pair, row 1 an active negative, row 2 an inactive negative.
    let x1 = Tensor::from_vec(vec![0.3, -1.2, 2.0, 0.7, 1.1, -0.4, 1.0, 0.5, -0.8], &[3, 3]).unwrap();
    let x2 = Tensor::from_vec(vec![1.5, -0.7, 0.2, 0.9, 0.4, -1.3, -1.0, 0.2, 0.9], &[3, 3]).unwrap();
    let target = Tensor::from_vec(vec![1.0, -1.0, -1.0], &[3]).unwrap();
    grad_check(&[x1, x2, target], |t| t[0].cosine_embedding_loss(&t[1], &t[2], 0.1, 1e-8).unwrap());
}
