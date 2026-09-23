use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec((0..n).map(|i| 0.15 + 0.27 * i as f32).collect::<Vec<_>>(), y.shape()).unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn rms_norm_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let y = a.rms_norm(None, 0.0).unwrap();
    assert_eq!(y.shape(), &[1, 3]);
    // rms([1,2,3]) = sqrt(14/3), and no mean is subtracted.
    let rms = (14.0f32 / 3.0).sqrt();
    let want = [1.0 / rms, 2.0 / rms, 3.0 / rms];
    for (g, w) in y.to_vec().iter().zip(&want) {
        assert!((g - w).abs() < 1e-5);
    }

    let w = Tensor::from_vec(vec![2.0, 0.5, -1.0], &[3]).unwrap();
    let y2 = a.rms_norm(Some(&w), 0.0).unwrap().to_vec();
    for (i, g) in y2.iter().enumerate() {
        assert!((g - want[i] * w.to_vec()[i]).abs() < 1e-5);
    }
}

#[test]
fn rms_norm_row_independence_and_eps() {
    let a = Tensor::from_vec(vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0], &[2, 3]).unwrap();
    let y = a.rms_norm(None, 1e-2).unwrap().to_vec();
    // Row 0: mean(x^2) = 25/3; row 1 is all zeros and must stay finite via eps.
    let inv = 1.0 / (25.0f32 / 3.0 + 1e-2).sqrt();
    assert!((y[0] - 3.0 * inv).abs() < 1e-5);
    assert!((y[1] - 4.0 * inv).abs() < 1e-5);
    assert!(y[2].abs() < 1e-7);
    for v in &y[3..] {
        assert!(v.is_finite() && v.abs() < 1e-7);
    }
}

#[test]
fn rms_norm_scale_invariance() {
    let a = Tensor::from_vec(vec![0.3, -1.2, 2.5, 0.7], &[2, 2]).unwrap();
    let scaled = a.mul(&Tensor::scalar(7.0)).unwrap();
    let y = a.rms_norm(None, 0.0).unwrap().to_vec();
    let ys = scaled.rms_norm(None, 0.0).unwrap().to_vec();
    for (p, q) in y.iter().zip(&ys) {
        assert!((p - q).abs() < 1e-5);
    }
}

#[test]
fn rms_norm_bad_shapes_are_err() {
    assert!(Tensor::scalar(1.0).rms_norm(None, 1e-5).is_err());
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let w = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(a.rms_norm(Some(&w), 1e-5).is_err());
}

#[test]
fn rms_norm_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, -0.5, 0.7, 1.3, -2.1], &[2, 3]).unwrap();
    grad_check(&[a], |t| weighted_loss(t[0].rms_norm(None, 1e-5).unwrap()));
}

#[test]
fn rms_norm_grad_with_weight() {
    let a = Tensor::from_vec(vec![1.0, 2.0, -0.5, 0.7, 1.3, -2.1], &[2, 3]).unwrap();
    let w = Tensor::from_vec(vec![0.8, -1.1, 1.4], &[3]).unwrap();
    grad_check(&[a, w], |t| weighted_loss(t[0].rms_norm(Some(&t[1]), 1e-5).unwrap()));
}
