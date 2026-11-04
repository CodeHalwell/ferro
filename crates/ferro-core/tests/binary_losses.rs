use ferro_core::{DType, Tensor};

#[test]
fn probability_boundaries_are_finite_and_use_the_clamped_adjoint() {
    let x = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4]).unwrap().requires_grad_(true).unwrap();
    let t = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], &[4]).unwrap();
    let loss = x.binary_cross_entropy_unreduced(&t).unwrap();
    assert_eq!(loss.to_vec(), vec![0.0, 0.0, 100.0, 100.0]);
    loss.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![0.0, 0.0, -1e12, 1e12]);
}

#[test]
fn float64_keeps_its_values_and_torch_float32_denominator_floor() {
    let x = Tensor::from_vec_f64(vec![0.0, 1e-30, 0.123456789123], &[3]).unwrap().requires_grad_(true).unwrap();
    let t = Tensor::from_vec_f64(vec![1.0, 0.2, 0.6], &[3]).unwrap().requires_grad_(true).unwrap();
    let loss = x.binary_cross_entropy_unreduced(&t).unwrap();
    assert_eq!(loss.dtype(), DType::F64);
    loss.sum().backward();
    let dx = x.grad().unwrap().to_vec_f64();
    assert_eq!(dx[0], -1.0 / (1e-12f32 as f64));
    let expected = (0.123456789123 - 0.6) / (0.123456789123 * (1.0 - 0.123456789123));
    assert!((dx[2] - expected).abs() < 1e-13);
    assert!(t.grad().unwrap().to_vec_f64()[0].is_infinite());
}

#[test]
fn weighted_logits_gradients_match_finite_differences() {
    let x = Tensor::from_vec(vec![-0.7, 0.0, 1.2, -1.1], &[2, 2]).unwrap();
    let t = Tensor::from_vec(vec![0.2, 0.9, 0.5, 0.7], &[2, 2]).unwrap();
    let positive = Tensor::from_vec(vec![0.4, 2.3], &[2]).unwrap();
    ferro_core::testkit::grad_check(&[x, t], |values| values[0].bce_with_logits_unreduced(&values[1], Some(&positive)).unwrap().sum());
}

#[test]
fn logits_zero_and_extremes_have_the_smooth_adjoint() {
    let x = Tensor::from_vec_f64(vec![-1000.0, 0.0, 1000.0], &[3]).unwrap().requires_grad_(true).unwrap();
    let t = Tensor::from_vec_f64(vec![0.0, 0.3, 1.0], &[3]).unwrap();
    let loss = x.bce_with_logits_unreduced(&t, None).unwrap();
    assert_eq!(loss.to_vec_f64(), vec![0.0, 2.0f64.ln(), 0.0]);
    loss.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec_f64(), vec![0.0, 0.2, 0.0]);
}

#[test]
fn validation_rejects_probability_domain_shape_and_differentiable_weights() {
    let x = Tensor::from_vec(vec![0.2, 0.7], &[2]).unwrap();
    assert!(x.binary_cross_entropy_unreduced(&Tensor::from_vec(vec![f32::NAN, 0.0], &[2]).unwrap()).is_err());
    assert!(x.binary_cross_entropy_unreduced(&Tensor::zeros(&[1, 2])).is_err());
    assert!(Tensor::from_vec(vec![-0.1, 0.5], &[2]).unwrap().binary_cross_entropy_unreduced(&x).is_err());
    let positive = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    assert!(x.bce_with_logits_unreduced(&x, Some(&positive)).is_err());
    assert!(x.bce_with_logits_unreduced(&x, Some(&Tensor::ones(&[2, 2]))).is_err());
}

#[test]
fn empty_losses_keep_their_layout_and_backward_edges() {
    let x = Tensor::from_vec_f64(vec![], &[0, 3]).unwrap().requires_grad_(true).unwrap();
    let t = Tensor::from_vec_f64(vec![], &[0, 3]).unwrap();
    let loss = x.bce_with_logits_unreduced(&t, None).unwrap();
    assert_eq!(loss.shape(), &[0, 3]);
    loss.sum().backward();
    assert_eq!(x.grad().unwrap().shape(), &[0, 3]);
}
