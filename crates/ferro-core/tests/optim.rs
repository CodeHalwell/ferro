use ferro_core::optim::{Adam, AdamW, Sgd};
use ferro_core::{Param, Rng, Tensor};

/// MSE between predictions and targets as a differentiable scalar loss.
fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.sub(target).unwrap();
    diff.mul(&diff).unwrap().mean()
}

/// Build a linear-regression problem `y = X @ w_true` with a random design
/// matrix. Returns (X, y, w_true) with `w_true` shaped `[in, 1]`.
fn linreg_problem(rows: usize, cols: usize, rng: &Rng) -> (Tensor, Tensor, Vec<f32>) {
    let x = Tensor::randn(&[rows, cols], rng);
    let w_true: Vec<f32> = (0..cols).map(|_| rng.normal()).collect();
    let wt = Tensor::from_vec(w_true.clone(), &[cols, 1]).unwrap();
    let y = x.matmul(&wt).unwrap().detach_copy();
    (x, y, w_true)
}

#[test]
fn sgd_fits_linear_regression() {
    let rng = Rng::new(0);
    let (x, y, w_true) = linreg_problem(64, 3, &rng);
    let w = Param::new(Tensor::randn(&[3, 1], &rng));
    let mut opt = Sgd::new(vec![w.clone()], 0.1);

    let initial = mse(&x.matmul(&w.tensor()).unwrap(), &y).item();
    for _ in 0..400 {
        let loss = mse(&x.matmul(&w.tensor()).unwrap(), &y);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    let final_loss = mse(&x.matmul(&w.tensor()).unwrap(), &y).item();
    assert!(
        final_loss < initial * 1e-3,
        "loss {final_loss} vs initial {initial}"
    );

    let learned = w.tensor().to_vec();
    for (i, &t) in w_true.iter().enumerate() {
        assert!(
            (learned[i] - t).abs() < 1e-2,
            "w[{i}] {} vs {t}",
            learned[i]
        );
    }
}

#[test]
fn sgd_momentum_fits_linear_regression() {
    let rng = Rng::new(1);
    let (x, y, w_true) = linreg_problem(64, 3, &rng);
    let w = Param::new(Tensor::randn(&[3, 1], &rng));
    let mut opt = Sgd::new(vec![w.clone()], 0.05).with_momentum(0.9);

    let initial = mse(&x.matmul(&w.tensor()).unwrap(), &y).item();
    for _ in 0..400 {
        let loss = mse(&x.matmul(&w.tensor()).unwrap(), &y);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    let final_loss = mse(&x.matmul(&w.tensor()).unwrap(), &y).item();
    assert!(
        final_loss < initial * 1e-3,
        "loss {final_loss} vs initial {initial}"
    );

    let learned = w.tensor().to_vec();
    for (i, &t) in w_true.iter().enumerate() {
        assert!(
            (learned[i] - t).abs() < 1e-2,
            "w[{i}] {} vs {t}",
            learned[i]
        );
    }
}

#[test]
fn adam_minimizes_quadratic() {
    let rng = Rng::new(2);
    let target = Tensor::from_vec(vec![1.5, -2.0, 0.75, 3.25], &[4]).unwrap();
    let w = Param::new(Tensor::randn(&[4], &rng));
    let mut opt = Adam::new(vec![w.clone()], 0.1);

    for _ in 0..500 {
        let loss = mse(&w.tensor(), &target);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    let learned = w.tensor().to_vec();
    for (i, t) in target.to_vec().iter().enumerate() {
        assert!(
            (learned[i] - t).abs() < 1e-3,
            "w[{i}] {} vs {t}",
            learned[i]
        );
    }
}

#[test]
fn adamw_minimizes_quadratic() {
    let rng = Rng::new(3);
    let target = Tensor::from_vec(vec![0.5, -1.0, 0.25], &[3]).unwrap();
    let w = Param::new(Tensor::randn(&[3], &rng));
    let mut opt = AdamW::new(vec![w.clone()], 0.1).with_weight_decay(0.0);

    for _ in 0..500 {
        let loss = mse(&w.tensor(), &target);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    let learned = w.tensor().to_vec();
    for (i, t) in target.to_vec().iter().enumerate() {
        assert!(
            (learned[i] - t).abs() < 1e-3,
            "w[{i}] {} vs {t}",
            learned[i]
        );
    }
}

#[test]
fn adamw_decay_is_decoupled_from_moments() {
    // One step from w=1 with dL/dw=0.5: Adam's update is exactly -lr (bias
    // correction cancels on the first step and eps is negligible); AdamW must
    // additionally shrink the parameter by lr*wd*w, and the decay must not
    // pass through the moment estimates.
    let make = || {
        let w = Param::new(Tensor::from_vec(vec![1.0], &[1]).unwrap());
        let loss = w.tensor().mul(&Tensor::scalar(0.5)).unwrap().sum();
        loss.backward();
        w
    };
    let (lr, wd) = (0.1, 0.04);

    let plain = make();
    AdamW::new(vec![plain.clone()], lr)
        .with_weight_decay(0.0)
        .step();
    let decayed = make();
    AdamW::new(vec![decayed.clone()], lr)
        .with_weight_decay(wd)
        .step();

    let p = plain.tensor().item();
    let d = decayed.tensor().item();
    assert!((p - (1.0 - lr)).abs() < 1e-4, "no-decay step: {p}");
    assert!(
        (d - (p - lr * wd * 1.0)).abs() < 1e-6,
        "decoupled decay: {d} vs {p}"
    );
}

#[test]
fn step_skips_param_without_grad() {
    let w = Param::new(Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap());
    let mut opt = Sgd::new(vec![w.clone()], 0.5);
    opt.step();
    assert_eq!(w.tensor().to_vec(), vec![1.0, 2.0]);
}

#[test]
fn adam_bias_correction_ignores_skipped_steps() {
    // Param b gets no gradient for the first k steps; when it finally receives
    // one, its update must match a fresh Adam's first step (per-param step
    // counts, not a global timestep).
    let data = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();

    let run = |skip_steps: usize| -> Vec<f32> {
        let a = ferro_core::Param::new(Tensor::from_vec(vec![0.5, -0.5], &[2, 1]).unwrap());
        let b = ferro_core::Param::new(Tensor::from_vec(vec![1.0, -2.0], &[2, 1]).unwrap());
        let mut opt = Adam::new(vec![a.clone(), b.clone()], 0.01);
        for _ in 0..skip_steps {
            let loss = data.matmul(&a.tensor()).unwrap().mean();
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
        let loss = data.matmul(&b.tensor()).unwrap().mean();
        opt.zero_grad();
        loss.backward();
        opt.step();
        b.tensor().to_vec()
    };

    assert_eq!(run(3), run(0));
}
