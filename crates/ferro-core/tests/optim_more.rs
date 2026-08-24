//! Tests for the optim.rs extensions: nesterov SGD hand-computed steps, AdamW
//! hand-computed step, convergence on tiny convex problems, scheduler closed
//! forms, and global-norm gradient clipping.

use ferro_core::optim::{
    global_grad_norm, AdamW, CosineWithWarmup, ExponentialLr, LrScheduler, Sgd, StepLr,
};
use ferro_core::{Param, Tensor};

fn param_of(v: &[f32]) -> Param {
    Param::new(Tensor::from_vec(v.to_vec(), &[v.len()]).unwrap())
}

/// Install an exact gradient on every param via the autograd engine:
/// `(param * g).sum()` has gradient exactly `g`.
fn set_grads(params: &[Param], g: &[f32]) {
    for p in params {
        let t = p.tensor();
        let shape = t.shape().to_vec();
        // Broadcast a scalar gradient across any param shape.
        let coef = if g.len() == 1 {
            Tensor::full(&shape, g[0])
        } else {
            assert_eq!(g.len(), t.numel());
            Tensor::from_vec(g.to_vec(), &shape).unwrap()
        };
        t.mul(&coef).unwrap().sum().backward();
    }
}

#[test]
fn sgd_nesterov_matches_hand_computed_steps() {
    // x' = x - lr*(g + mu*(mu*v + g)) with v0 = 0.
    let p = param_of(&[1.0]);
    let mut opt = Sgd::new(vec![p.clone()], 0.1)
        .with_momentum(0.9)
        .with_nesterov(true);
    set_grads(&[p.clone()], &[1.0]);

    opt.step();
    // v1 = 1; update = g + mu*v = 1 + 0.9 = 1.9 => x = 1 - 0.19 = 0.81
    assert!((p.tensor().to_vec()[0] - 0.81).abs() < 1e-6);

    set_grads(&[p.clone()], &[1.0]);
    opt.step();
    // v2 = 0.9*1+1 = 1.9; update = 1 + 0.9*1.9 = 2.71 => x = 0.81 - 0.271
    assert!((p.tensor().to_vec()[0] - 0.539).abs() < 1e-5);
}

#[test]
fn sgd_plain_momentum_unchanged_by_nesterov_flag_off() {
    let p = param_of(&[1.0]);
    let mut opt = Sgd::new(vec![p.clone()], 0.1).with_momentum(0.9);
    set_grads(&[p.clone()], &[1.0]);
    opt.step();
    assert!((p.tensor().to_vec()[0] - 0.9).abs() < 1e-6); // lr * v = 0.1
}

#[test]
fn adamw_first_step_matches_hand_computed_value() {
    // Defaults wd=0.01: m_hat = g, v_hat = g^2, update =
    // -lr*(g/(|g|+eps) + wd*x).
    let p = param_of(&[2.0]);
    let mut opt = AdamW::new(vec![p.clone()], 0.01).with_weight_decay(0.1);
    set_grads(&[p.clone()], &[4.0]);
    opt.step();
    let eps = 1e-8;
    let want = 2.0 - 0.01 * (4.0 / (16.0f32.sqrt() + eps) + 0.1 * 2.0);
    assert!(
        (p.tensor().to_vec()[0] - want).abs() < 1e-5,
        "got {} want {want}",
        p.tensor().to_vec()[0]
    );
}

#[test]
fn optimizers_converge_on_tiny_convex_problem() {
    // Minimize f(x) = (x-3)^2 for a scalar param; grad = 2(x-3).
    fn run(mut opt: impl FnMut(&Param)) -> f32 {
        let p = param_of(&[0.0]);
        for _ in 0..300 {
            let x = p.tensor().to_vec()[0];
            set_grads(&[p.clone()], &[2.0 * (x - 3.0)]);
            opt(&p.clone());
        }
        p.tensor().to_vec()[0]
    }
    let x = run(|p| {
        let mut o = Sgd::new(vec![p.clone()], 0.05)
            .with_momentum(0.9)
            .with_nesterov(true);
        o.step();
    });
    assert!((x - 3.0).abs() < 1e-2, "sgd nesterov landed at {x}");

    let x = run(|p| {
        let mut o = AdamW::new(vec![p.clone()], 0.1).with_weight_decay(0.0);
        o.step();
    });
    assert!((x - 3.0).abs() < 1e-2, "adamw landed at {x}");
}

#[test]
fn global_norm_and_clipping_math() {
    let a = param_of(&[0.0, 0.0]);
    let b = param_of(&[0.0]);
    set_grads(&[a.clone()], &[3.0, 4.0]); // norm 5
    set_grads(&[b.clone()], &[12.0]);
    assert!((global_grad_norm(&[a.clone(), b.clone()]) - 13.0).abs() < 1e-4);

    // Under budget: no clipping applied by the optimizer step.
    a.zero_grad();
    b.zero_grad();
    let mut opt = Sgd::new(vec![a.clone(), b.clone()], 1.0).with_max_grad_norm(100.0);
    set_grads(&[a.clone(), b.clone()], &[1.0]);
    opt.step();
    assert_eq!(a.tensor().to_vec(), vec![-1.0, -1.0]);

    // Over budget: the effective gradient is rescaled to norm max.
    let c = param_of(&[10.0, 10.0]);
    let mut clip = AdamW::new(vec![c.clone()], 1.0)
        .with_weight_decay(0.0)
        .with_max_grad_norm(1.0);
    set_grads(&[c.clone()], &[30.0, 40.0]); // norm 50 -> scale 1/50
    clip.step();
    // Clipped grads are [0.6, 0.8]; Adam's first step with lr=1 moves each
    // coordinate by ~g/(|g|+eps) = sign, so both land near 9.
    let v = c.tensor().to_vec();
    assert!(
        (v[0] - 9.0).abs() < 1e-3 && (v[1] - 9.0).abs() < 1e-3,
        "{v:?}"
    );
}

#[test]
fn schedulers_match_closed_forms() {
    let step_lr = StepLr {
        base_lr: 0.4,
        step_size: 10,
        gamma: 0.5,
    };
    assert_eq!(step_lr.lr(0), 0.4);
    assert_eq!(step_lr.lr(9), 0.4);
    assert_eq!(step_lr.lr(10), 0.2);
    assert_eq!(step_lr.lr(25), 0.4 * 0.5f32.powi(2));

    let exp = ExponentialLr {
        base_lr: 1.0,
        gamma: 0.9,
    };
    assert_eq!(exp.lr(0), 1.0);
    assert!((exp.lr(3) - 0.729).abs() < 1e-6);

    let cos = CosineWithWarmup {
        base_lr: 1.0,
        min_lr: 0.0,
        warmup_steps: 10,
        total_steps: 110,
    };
    assert_eq!(cos.lr(0), 0.0);
    assert!((cos.lr(5) - 0.5).abs() < 1e-6);
    assert_eq!(cos.lr(10), 1.0);
    // Halfway through the cosine phase (step 60 of 110): cos(pi/2)=0.
    assert!((cos.lr(60) - 0.5).abs() < 1e-6);
    // Past total: held at min_lr.
    assert_eq!(cos.lr(200), 0.0);

    // Non-zero floor.
    let cos2 = CosineWithWarmup {
        base_lr: 1.0,
        min_lr: 0.1,
        warmup_steps: 0,
        total_steps: 100,
    };
    assert!((cos2.lr(50) - 0.55).abs() < 1e-6);
    assert_eq!(cos2.lr(500), 0.1);
}
