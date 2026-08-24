use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.11 + 0.37 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn group_norm_values() {
    // 1 group over [2, 4]: the whole row is one statistic.
    let a = Tensor::randn(&[1, 4], &ferro_core::Rng::new(9));
    let w = Tensor::ones(&[4]);
    let b = Tensor::zeros(&[4]);
    let y = a.group_norm(1, &w, &b, 1e-5).unwrap().to_vec();
    let m = y.iter().sum::<f32>() / 4.0;
    let v = y.iter().map(|v| (v - m).powi(2)).sum::<f32>() / 4.0;
    assert!(m.abs() < 1e-5);
    assert!((v - 1.0).abs() < 1e-2);

    // 2 groups on rank-2: each half normalized independently, then affine.
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let wv = vec![2.0, 2.0, -1.0, -1.0];
    let bv = vec![0.1, 0.1, 0.2, 0.2];
    let w = Tensor::from_vec(wv.clone(), &[4]).unwrap();
    let b = Tensor::from_vec(bv.clone(), &[4]).unwrap();
    let y = a.group_norm(2, &w, &b, 1e-5).unwrap().to_vec();
    // Each two-element group {1,2} / {3,4} has biased std 0.5, so xhat is +/-1.
    for (i, want) in [(0usize, -1.0), (1, 1.0), (2, -1.0), (3, 1.0)] {
        assert!(
            (y[i] - (want * wv[i] + bv[i])).abs() < 1e-3,
            "elem {i} got {}",
            y[i]
        );
    }
}

#[test]
fn group_norm_errors() {
    let a = Tensor::zeros(&[2, 4]);
    let w = Tensor::ones(&[4]);
    let b = Tensor::zeros(&[4]);
    assert!(a.group_norm(3, &w, &b, 1e-5).is_err());
    assert!(a.group_norm(0, &w, &b, 1e-5).is_err());
    let w_bad = Tensor::ones(&[3]);
    assert!(a.group_norm(2, &w_bad, &b, 1e-5).is_err());
    assert!(Tensor::zeros(&[4]).group_norm(1, &w, &b, 1e-5).is_err());
}

#[test]
fn group_norm_grad_rank2() {
    let a = Tensor::randn(&[2, 4], &ferro_core::Rng::new(13));
    let w = Tensor::from_vec(vec![1.1, 0.7, -0.6, 0.9], &[4]).unwrap();
    let b = Tensor::from_vec(vec![0.2, -0.4, 0.3, -0.1], &[4]).unwrap();
    grad_check(&[a.clone(), w.clone(), b.clone()], |t| {
        weighted_loss(t[0].group_norm(2, &t[1], &t[2], 1e-5).unwrap())
    });
    grad_check(&[a], |t| {
        weighted_loss(
            t[0].group_norm(1, &Tensor::ones(&[4]), &Tensor::zeros(&[4]), 1e-5)
                .unwrap(),
        )
    });
}

#[test]
fn group_norm_grad_rank4() {
    let a = Tensor::randn(&[2, 2, 2, 2], &ferro_core::Rng::new(17));
    let w = Tensor::ones(&[2]);
    let b = Tensor::zeros(&[2]);
    grad_check(&[a, w, b], |t| {
        weighted_loss(t[0].group_norm(1, &t[1], &t[2], 1e-5).unwrap())
    });
}
