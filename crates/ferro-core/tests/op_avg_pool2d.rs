use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.17 + 0.23 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn avg_pool2d_values() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
    let y = x.avg_pool2d(2, 2).unwrap();
    assert_eq!(y.shape(), &[1, 1, 1, 1]);
    assert!((y.item() - 2.5).abs() < 1e-6);

    // Stride 1 with 2x2 kernel on a 1x3x3 plane: overlapping windows.
    let x = Tensor::from_vec((1..=9).map(|v| v as f32).collect(), &[1, 1, 3, 3]).unwrap();
    let y = x.avg_pool2d(2, 1).unwrap();
    assert_eq!(y.shape(), &[1, 1, 2, 2]);
    assert_eq!(y.to_vec(), vec![3.0, 4.0, 6.0, 7.0]);
}

#[test]
fn avg_pool2d_errors() {
    let x = Tensor::zeros(&[1, 1, 2, 2]);
    assert!(x.avg_pool2d(3, 1).is_err());
    assert!(x.avg_pool2d(0, 1).is_err());
    assert!(Tensor::zeros(&[4]).avg_pool2d(2, 2).is_err());
}

#[test]
fn avg_pool2d_grad() {
    // Overlapping windows accumulate gradients; O(1)-magnitude inputs.
    let x = Tensor::from_vec(
        vec![0.6, -1.1, 0.4, 0.9, -0.3, 1.2, 0.5, -0.8, 0.2],
        &[1, 1, 3, 3],
    )
    .unwrap();
    grad_check(&[x.clone()], |t| {
        weighted_loss(t[0].avg_pool2d(2, 1).unwrap())
    });

    // Non-overlapping, stride 2.
    let x = Tensor::from_vec(vec![0.7, -0.5, 1.1, -0.2], &[1, 1, 2, 2]).unwrap();
    grad_check(&[x], |t| weighted_loss(t[0].avg_pool2d(2, 2).unwrap()));
}
