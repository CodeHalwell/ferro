use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.15 + 0.27 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn layer_norm_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let y = a.layer_norm(None, None, 1e-5).unwrap();
    // var([1,2,3]) = 2/3 biased, so xhat = +/-sqrt(3/2).
    let s = (3.0f32 / 2.0).sqrt();
    let want = vec![-s, 0.0, s];
    for (g, w) in y.to_vec().iter().zip(&want) {
        assert!((g - w).abs() < 1e-5);
    }

    let w = Tensor::from_vec(vec![2.0, 0.5, -1.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![0.1, -0.2, 0.3], &[3]).unwrap();
    let y2 = a.layer_norm(Some(&w), Some(&b), 1e-5).unwrap().to_vec();
    for (i, g) in y2.iter().enumerate() {
        assert!((g - (want[i] * w.to_vec()[i] + b.to_vec()[i])).abs() < 1e-3);
    }
}

#[test]
fn layer_norm_row_independence() {
    let a = Tensor::from_vec(vec![10.0, 20.0, 30.0, -5.0, 0.0, 5.0], &[2, 3]).unwrap();
    let y = a.layer_norm(None, None, 1e-5).unwrap().to_vec();
    let s = (3.0f32 / 2.0).sqrt();
    for (row, base) in [(0usize, 0usize), (1, 3)] {
        assert!((y[base] + s).abs() < 1e-3);
        assert!(y[base + 1].abs() < 1e-3);
        assert!((y[base + 2] - s).abs() < 1e-3);
        let _ = row;
    }
}

#[test]
fn layer_norm_errors() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let w = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(a.layer_norm(Some(&w), None, 1e-5).is_err());
}

#[test]
fn layer_norm_grad() {
    let a = Tensor::from_vec(vec![0.7, -1.2, 0.3, 1.4, -0.5, 0.9], &[2, 3]).unwrap();
    let w = Tensor::from_vec(vec![1.1, 0.6, -0.8], &[3]).unwrap();
    let b = Tensor::from_vec(vec![0.2, -0.3, 0.5], &[3]).unwrap();

    grad_check(&[a.clone()], |t| {
        weighted_loss(t[0].layer_norm(None, None, 1e-5).unwrap())
    });
    grad_check(&[a.clone(), w.clone()], |t| {
        weighted_loss(t[0].layer_norm(Some(&t[1]), None, 1e-5).unwrap())
    });
    grad_check(&[a.clone(), w.clone(), b.clone()], |t| {
        weighted_loss(t[0].layer_norm(Some(&t[1]), Some(&t[2]), 1e-5).unwrap())
    });
}
