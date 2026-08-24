use ferro_core::{testkit::grad_check, Rng, Tensor};

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.13 + 0.29 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

fn params(c: usize) -> (Tensor, Tensor, Tensor, Tensor) {
    let wv: Vec<f32> = (0..c).map(|i| 0.9 - 0.25 * i as f32).collect();
    let bv: Vec<f32> = (0..c).map(|i| -0.2 + 0.3 * i as f32).collect();
    (
        Tensor::from_vec(wv, &[c]).unwrap(),
        Tensor::from_vec(bv, &[c]).unwrap(),
        Tensor::zeros(&[c]),
        Tensor::ones(&[c]),
    )
}

#[test]
fn batch_norm_train_normalizes_per_channel() {
    let x = Tensor::randn(&[4, 3], &Rng::new(7));
    let (w, b, rm, rv) = params(3);
    let out = x
        .batch_norm(&w, &b, &rm, &rv, 1e-5, true, 0.1)
        .unwrap()
        .output
        .to_vec();
    for chn in 0..3 {
        let col: Vec<f32> = (0..4).map(|r| out[r * 3 + chn]).collect();
        let m = col.iter().sum::<f32>() / 4.0;
        let v = col
            .iter()
            .map(|v| (v - b.to_vec()[chn]).powi(2))
            .sum::<f32>()
            / 4.0;
        assert!((m - b.to_vec()[chn]).abs() < 1e-4, "channel {chn} mean {m}");
        assert!(
            (v - w.to_vec()[chn].powi(2)).abs() < 1e-2,
            "channel {chn} var {v}"
        );
    }
}

#[test]
fn batch_norm_running_stats_update() {
    let x = Tensor::randn(&[8, 2], &Rng::new(11));
    let (w, b, rm, rv) = params(2);
    let res = x.batch_norm(&w, &b, &rm, &rv, 1e-5, true, 0.1).unwrap();
    for chn in 0..2 {
        let col: Vec<f32> = x.to_vec()[chn..].iter().step_by(2).cloned().collect();
        let mu = col.iter().sum::<f32>() / 8.0;
        let v = col.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / 7.0;
        let want_m = 0.9 * rm.to_vec()[chn] + 0.1 * mu;
        let want_v = 0.9 * rv.to_vec()[chn] + 0.1 * v;
        assert!((res.running_mean.to_vec()[chn] - want_m).abs() < 1e-5);
        assert!((res.running_var.to_vec()[chn] - want_v).abs() < 1e-4);
    }
}

#[test]
fn batch_norm_eval_uses_running_stats() {
    let x = Tensor::randn(&[3, 2], &Rng::new(3));
    let rm = Tensor::from_vec(vec![0.5, -0.5], &[2]).unwrap();
    let rv = Tensor::from_vec(vec![2.0, 0.5], &[2]).unwrap();
    let (w, _b, _rm0, _rv0) = params(2);
    let b = Tensor::zeros(&[2]);
    let out = x
        .batch_norm(&w, &b, &rm, &rv, 1e-5, false, 0.1)
        .unwrap()
        .output
        .to_vec();
    let xv = x.to_vec();
    for chn in 0..2 {
        for r in 0..3 {
            let want = ((xv[r * 2 + chn] - rm.to_vec()[chn]) / (rv.to_vec()[chn] + 1e-5).sqrt())
                * w.to_vec()[chn];
            assert!((out[r * 2 + chn] - want).abs() < 1e-5);
        }
    }
}

#[test]
fn batch_norm_errors() {
    let x = Tensor::from_vec(vec![0.0; 6], &[2, 3]).unwrap();
    let (_, _, rm, rv) = params(3);
    let w_bad = Tensor::from_vec(vec![1.0], &[1]).unwrap();
    let b = Tensor::zeros(&[3]);
    assert!(x.batch_norm(&w_bad, &b, &rm, &rv, 1e-5, true, 0.1).is_err());
    let x3 = Tensor::from_vec(vec![0.0; 8], &[2, 2, 2]).unwrap();
    assert!(x3
        .batch_norm(&Tensor::ones(&[2]), &b, &rm, &rv, 1e-5, true, 0.1)
        .is_err());
}

#[test]
fn batch_norm_grad_rank2() {
    let x = Tensor::randn(&[4, 3], &Rng::new(21));
    let (w, b, rm, rv) = params(3);
    grad_check(&[x.clone(), w.clone(), b.clone()], |t| {
        weighted_loss(
            t[0].batch_norm(&t[1], &t[2], &rm, &rv, 1e-5, true, 0.1)
                .unwrap()
                .output,
        )
    });
}

#[test]
fn batch_norm_grad_rank4() {
    let x = Tensor::randn(&[2, 2, 2, 2], &Rng::new(33));
    let (w, b, rm, rv) = params(2);
    grad_check(&[x, w, b], |t| {
        weighted_loss(
            t[0].batch_norm(&t[1], &t[2], &rm, &rv, 1e-5, true, 0.1)
                .unwrap()
                .output,
        )
    });
}
