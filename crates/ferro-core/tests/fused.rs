use ferro_core::fused_ops::Act;
use ferro_core::testkit::{grad_check, grad_check_strict};
use ferro_core::Tensor;

#[test]
fn bias_add_gelu_values_match_unfused() {
    let x = Tensor::from_vec(vec![-1.0, 0.5, 2.0, -0.3, 1.5, -2.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![0.1, -0.2, 0.3], &[3]).unwrap();
    let fused = x.bias_add_activation(&b, Act::Gelu).unwrap().to_vec();
    let z = x.add(&b).unwrap();
    let want = z.gelu().to_vec();
    for (g, w) in fused.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
    }
}

#[test]
fn bias_add_relu_and_silu_values_match_unfused() {
    let x = Tensor::from_vec(vec![-1.0, 0.5, 2.0, -0.3, 1.5, -2.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![0.4, -0.6, 0.1], &[3]).unwrap();
    for act in [Act::Relu, Act::Silu, Act::Identity] {
        let fused = x.bias_add_activation(&b, act).unwrap().to_vec();
        let z = x.add(&b).unwrap();
        let want = match act {
            Act::Relu => z.relu(),
            Act::Silu => z.silu(),
            Act::Identity => z,
            Act::Gelu => unreachable!(),
        }
        .to_vec();
        for (g, w) in fused.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "{act:?}: got {g}, want {w}");
        }
    }
}

#[test]
fn bias_add_activation_grads() {
    // Bias chosen so every pre-activation sits away from the relu kink at 0.
    let x = Tensor::from_vec(vec![-1.7, 0.9, 2.3, -0.8, 1.4, -2.6], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![0.5, 0.3, 0.7], &[3]).unwrap();
    grad_check_strict(&[x, b], |t| {
        t[0].bias_add_activation(&t[1], Act::Relu).unwrap().sum()
    });
}

#[test]
fn bias_add_gelu_grad() {
    let x = Tensor::from_vec(vec![0.4, -1.1, 1.9, 0.7, -0.5, 1.2], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![0.2, 0.4, -0.3], &[3]).unwrap();
    grad_check_strict(&[x, b], |t| {
        t[0].bias_add_activation(&t[1], Act::Gelu).unwrap().sum()
    });
}

#[test]
fn residual_layernorm_value_matches_composition() {
    let x = Tensor::randn(&[3, 4], &ferro_core::Rng::new(42));
    let r = Tensor::randn(&[3, 4], &ferro_core::Rng::new(43));
    let w = Tensor::from_vec(vec![1.1, 0.9, 1.05, 0.95], &[4]).unwrap();
    let b = Tensor::from_vec(vec![0.1, -0.1, 0.05, -0.05], &[4]).unwrap();
    let eps = 1e-5;
    let fused = x
        .residual_layernorm(&r, Some(&w), Some(&b), eps)
        .unwrap()
        .to_vec();
    let s = x.add(&r).unwrap();
    let want = s.layer_norm(Some(&w), Some(&b), eps).unwrap().to_vec();
    for (g, wv) in fused.iter().zip(want) {
        assert!((g - wv).abs() < 1e-5, "got {g}, want {wv}");
    }
}

#[test]
fn residual_layernorm_grads_all_operands() {
    let x = Tensor::randn(&[2, 4], &ferro_core::Rng::new(1));
    let r = Tensor::randn(&[2, 4], &ferro_core::Rng::new(2));
    let w = Tensor::ones(&[4]);
    let b = Tensor::zeros(&[4]);
    grad_check_strict(&[x, r, w, b], |t| {
        t[0].residual_layernorm(&t[1], Some(&t[2]), Some(&t[3]), 1e-5)
            .unwrap()
            .sum()
    });
}

#[test]
fn residual_layernorm_without_affine_grads() {
    let x = Tensor::randn(&[2, 4], &ferro_core::Rng::new(5));
    let r = Tensor::randn(&[2, 4], &ferro_core::Rng::new(6));
    grad_check_strict(&[x, r], |t| {
        t[0].residual_layernorm(&t[1], None, None, 1e-5)
            .unwrap()
            .sum()
    });
}
