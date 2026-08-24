//! Tests for the nn extensions in src/modules.rs plus init schemes and
//! train/eval mode plumbing: param discovery counts, mode switching behaviour,
//! conv2d/batchnorm numerics, and init statistics.

use ferro_core::modules::{BatchNorm, Conv2D, Dropout, ModuleList};
use ferro_core::nn::{eval, Init, Linear, Module, Relu, Sequential};
use ferro_core::{nn, Rng, Tensor};

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

fn var(v: &[f32]) -> f32 {
    let m = mean(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32
}

#[test]
fn param_discovery_counts_and_names() {
    // Linear 2, LayerNorm 2, Embedding 1 => 5 params.
    let seq = Sequential::new(vec![
        Box::new(Linear::new(4, 8, &Rng::new(1))),
        Box::new(nn::LayerNorm::new(8)),
        Box::new(nn::Embedding::new(10, 4, &Rng::new(11))),
    ]);
    let names: Vec<String> = seq.named_parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(
        names,
        vec!["0.weight", "0.bias", "1.weight", "1.bias", "2.weight"]
    );
    assert_eq!(seq.parameters().len(), 5);

    // Nested containers compose prefixes; Conv2D adds weight + bias.
    let outer = Sequential::new(vec![
        Box::new(seq),
        Box::new(Conv2D::new(3, 4, 3, &Rng::new(2))),
    ]);
    let names: Vec<String> = outer
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names.len(), 7);
    assert_eq!(names[5], "1.weight");
    assert_eq!(names[6], "1.bias");
}

#[test]
fn module_list_container() {
    let list = ModuleList::new(vec![
        Box::new(Linear::new(2, 3, &Rng::new(3))),
        Box::new(Relu),
    ]);
    assert_eq!(list.len(), 2);
    assert!(!list.is_empty());
    assert_eq!(list.named_parameters().len(), 2);
    let x = Tensor::from_vec(vec![1.0, -2.0, 0.5, 0.0], &[2, 2]).unwrap();
    let y = list.forward(&x).unwrap();
    assert_eq!(y.shape(), &[2, 3]);
}

#[test]
fn dropout_train_eval_switch() {
    let d = Dropout::new(0.5).with_seed(42);
    let x = Tensor::full(&[4, 50], 1.0);
    let out = d.forward(&x).unwrap().to_vec();
    // Train: some zeros, survivors scaled by 1/(1-p) = 2.
    assert!(out.iter().any(|v| *v == 0.0));
    for v in out.iter().filter(|v| **v != 0.0) {
        assert!((v - 2.0).abs() < 1e-6);
    }
    let kept = out.iter().filter(|v| **v != 0.0).count();
    assert!(kept > 60 && kept < 140, "kept {kept} of {}", out.len());

    // Eval: exact identity.
    eval(&d);
    assert_eq!(d.forward(&x).unwrap().to_vec(), x.to_vec());
}

#[test]
fn sequential_propagates_training_mode() {
    let seq = Sequential::new(vec![
        Box::new(Dropout::new(0.9)),
        Box::new(Linear::new(8, 8, &Rng::new(4))),
    ]);
    let x = Tensor::full(&[4, 8], 1.0);
    assert!(seq.forward(&x).unwrap().to_vec().iter().any(|v| *v == 0.0));
    eval(&seq);
    assert!(!seq.forward(&x).unwrap().to_vec().iter().any(|v| *v == 0.0));
    nn::train(&seq);
    assert!(seq.forward(&x).unwrap().to_vec().iter().any(|v| *v == 0.0));
}

#[test]
fn conv2d_layer_known_values_and_grads() {
    // 1 input/output channel, 1x1 kernel set directly so forward is exactly
    // y = w*x + b elementwise.
    let mut conv = Conv2D::with_config(1, 1, 1, 1, 0, &Rng::new(5));
    let params = conv.named_parameters();
    params[0]
        .1
        .set(Tensor::from_vec(vec![2.0], &[1, 1, 1, 1]).unwrap());
    params[1].1.set(Tensor::from_vec(vec![3.0], &[1]).unwrap());
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.shape(), &[1, 1, 2, 2]);
    assert_eq!(y.to_vec(), vec![5.0, 7.0, 9.0, 11.0]);

    // Gradient flows to weight and bias through a scalar loss:
    // dL/dW = sum(x) = 10, dL/db = numel = 4.
    y.sum().backward();
    let grads: Vec<f32> = conv
        .named_parameters()
        .iter()
        .map(|(_, p)| p.grad().unwrap().to_vec()[0])
        .collect();
    assert!((grads[0] - 10.0).abs() < 1e-4, "grads {grads:?}");
    assert!((grads[1] - 4.0).abs() < 1e-4);

    // Padding/stride output-shape formula on a bigger case.
    let conv2 = Conv2D::with_config(2, 4, 3, 2, 1, &Rng::new(6));
    let big = Tensor::randn(&[2, 2, 7, 7], &Rng::new(7));
    let out = conv2.forward(&big).unwrap();
    assert_eq!(out.shape(), &[2, 4, 4, 4]);
}

#[test]
fn batchnorm_train_normalizes_and_gradients_flow() {
    let bn = BatchNorm::new(3);
    let x = Tensor::from_vec(
        vec![
            1.0, 10.0, 100.0, //
            3.0, 14.0, 108.0,
        ],
        &[2, 3],
    )
    .unwrap();
    let out = bn.forward(&x).unwrap();
    // Feature 0: mean 2, biased var 1 -> normalized [-1, 1]. Feature 1:
    // mean 12, var 4 -> also [-1, 1].
    for (neg, pos) in [(0usize, 3usize), (1, 4)] {
        let v = out.to_vec();
        assert!(
            (v[neg] + 1.0).abs() < 1e-4 && (v[pos] - 1.0).abs() < 1e-4,
            "v {v:?}"
        );
    }

    // Gradients reach gamma/beta through a scalar loss.
    out.sum().backward();
    let g: Vec<Option<f32>> = bn
        .named_parameters()
        .into_iter()
        .map(|(_, p)| p.grad().map(|g| g.to_vec()[0]))
        .collect();
    assert!(
        g[0].is_some() && g[1].is_some(),
        "gamma/beta got no grad: {g:?}"
    );
}

#[test]
fn batchnorm_eval_uses_running_stats_not_batch_stats() {
    let bn = BatchNorm::new(2);
    // Feed training batches whose stats drift upward.
    for k in 0..60 {
        let x = Tensor::full(&[2, 2], k as f32);
        let _ = bn.forward(&x).unwrap();
    }
    eval(&bn);
    // Probe at zero: running mean is far positive, running var ~1, so the
    // eval output must be negative - batch-stat normalization of zeros would
    // give exactly 0.
    let probe = Tensor::zeros(&[1, 2]);
    let out = bn.forward(&probe).unwrap().to_vec();
    assert!(
        out.iter().all(|v| *v < -10.0),
        "eval should use running stats: {out:?}"
    );
    // Deterministic across calls.
    assert_eq!(bn.forward(&probe).unwrap().to_vec(), out);
}

#[test]
fn linear_with_init_schemes_sample_right_std() {
    assert!((Init::Kaiming.std(64, 64) - (2.0f32 / 64.0).sqrt()).abs() < 1e-6);
    assert!((Init::Xavier.std(64, 16) - (2.0f32 / 80.0).sqrt()).abs() < 1e-6);
    assert_eq!(Init::Normal(0.05).std(64, 64), 0.05);

    for (init, want) in [
        (Init::Kaiming, (2.0f32 / 128.0).sqrt()),
        (Init::Xavier, (2.0f32 / 256.0).sqrt()),
        (Init::Normal(0.1), 0.1),
    ] {
        let t = init.fill(&Rng::new(8), &[2000, 128], 128, 128);
        let v = t.to_vec();
        let sd = var(&v).sqrt();
        assert!(
            (sd - want).abs() < 0.02 * want + 0.002,
            "{init:?}: std {sd} vs {want}"
        );
        assert!(mean(&v).abs() < 0.01);
    }

    // Linear::with_init wires a scheme into the layer's weight.
    let lin = Linear::with_init(64, 64, &Rng::new(9), Init::Xavier);
    let w = lin.named_parameters()[0].1.tensor().to_vec();
    let want = (2.0f32 / 128.0).sqrt();
    assert!((var(&w).sqrt() - want).abs() < 0.02 * want);
}
