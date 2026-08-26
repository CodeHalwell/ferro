//! Grouped-query attention and learned positional embeddings - the two
//! architecture pieces real checkpoints need beyond the LLaMA-1-shaped MHA.

use ferro_core::nn::{LearnedPositionalEmbedding, Module, MultiHeadAttention};
use ferro_core::{Param, Rng, Tensor};

fn input(b: usize, s: usize, d: usize, seed: u64) -> Tensor {
    Tensor::randn(&[b, s, d], &Rng::new(seed))
}

#[test]
fn gqa_shapes_and_parameter_shapes() {
    let rng = Rng::new(3);
    let attn = MultiHeadAttention::with_kv_heads(8, 4, 2, true, &rng).unwrap();
    let params = attn.named_parameters();
    let shape_of = |name: &str| -> Vec<usize> {
        params
            .iter()
            .find(|(n, _)| n == name)
            .unwrap()
            .1
            .tensor()
            .shape()
            .to_vec()
    };
    // head_dim = 2; k/v project to kv_heads * head_dim = 4 columns (the HF
    // checkpoint shape), q/o stay square.
    assert_eq!(shape_of("q_proj"), vec![8, 8]);
    assert_eq!(shape_of("k_proj"), vec![8, 4]);
    assert_eq!(shape_of("v_proj"), vec![8, 4]);
    assert_eq!(shape_of("o_proj"), vec![8, 8]);

    let y = attn.forward(&input(2, 5, 8, 4)).unwrap();
    assert_eq!(y.shape(), &[2, 5, 8]);
}

#[test]
fn mqa_and_full_heads_degenerate_cases_work() {
    let rng = Rng::new(5);
    // kv_heads = 1: multi-query attention.
    let mqa = MultiHeadAttention::with_kv_heads(6, 3, 1, true, &rng).unwrap();
    assert_eq!(mqa.forward(&input(1, 4, 6, 6)).unwrap().shape(), &[1, 4, 6]);
    // Invalid group splits are rejected.
    assert!(MultiHeadAttention::with_kv_heads(8, 4, 3, true, &rng).is_err());
    assert!(MultiHeadAttention::with_kv_heads(8, 4, 0, true, &rng).is_err());
}

#[test]
fn kv_equal_heads_is_exactly_the_old_mha() {
    // new() delegates to with_kv_heads(heads, heads); same rng stream must
    // give identical weights and outputs.
    let a = MultiHeadAttention::new(8, 4, true, &Rng::new(9)).unwrap();
    let b = MultiHeadAttention::with_kv_heads(8, 4, 4, true, &Rng::new(9)).unwrap();
    let x = input(2, 3, 8, 10);
    assert_eq!(a.forward(&x).unwrap().to_vec(), b.forward(&x).unwrap().to_vec());
}

#[test]
fn grouped_kv_grads_sum_over_the_query_group() {
    // Analytic anchor: with q identically 0, softmax over causal scores is uniform, so
    // the output is a running mean of v rows - and dL/d(k_proj) must be
    // exactly zero (scores' k-gradient scales by q), while dL/d(v_proj) is
    // nonzero and finite. This exercises the expansion's backward: each kv
    // head accumulates from group = heads/kv_heads query heads.
    let rng = Rng::new(11);
    let attn = MultiHeadAttention::with_kv_heads(4, 4, 2, true, &rng).unwrap();
    // Zero the q projection so the anchor holds.
    let q = attn
        .named_parameters()
        .into_iter()
        .find(|(n, _)| n == "q_proj")
        .unwrap()
        .1;
    q.set(Tensor::zeros(&[4, 4]));

    let x = input(1, 3, 4, 12);
    let loss = attn.forward(&x).unwrap().sum();
    loss.backward();

    let grad_of = |name: &str| -> Vec<f32> {
        attn.named_parameters()
            .into_iter()
            .find(|(n, _)| n == name)
            .unwrap()
            .1
            .grad()
            .expect("param got a gradient")
            .to_vec()
    };
    let gk = grad_of("k_proj");
    assert!(gk.iter().all(|g| g.abs() < 1e-6), "k grad should vanish: {gk:?}");
    let gv = grad_of("v_proj");
    assert!(gv.iter().any(|g| g.abs() > 1e-3), "v grad must flow: {gv:?}");
    assert!(gv.iter().all(|g| g.is_finite()));
}

#[test]
fn gqa_grad_check_end_to_end() {
    // Finite-difference check through projection -> rope -> expand -> sdpa.
    let rng = Rng::new(13);
    let attn = MultiHeadAttention::with_kv_heads(4, 2, 1, true, &rng).unwrap().with_rope(10000.0);
    let x = Tensor::randn(&[1, 3, 4], &Rng::new(14));
    ferro_core::testkit::grad_check(&[x], |t| attn.forward(&t[0]).unwrap().sum());
}

#[test]
fn learned_positions_add_rows_and_accumulate_grads() {
    let rng = Rng::new(21);
    let pos = LearnedPositionalEmbedding::new(8, 3, &rng);
    let w = pos.named_parameters()[0].1.clone();

    let x = Tensor::zeros(&[2, 2, 3]);
    let y = pos.forward(&x).unwrap();
    // Every batch row gets the same first rows of the table.
    let table = w.tensor().to_vec();
    let out = y.to_vec();
    assert_eq!(&out[..6], &table[..6]);
    assert_eq!(&out[6..12], &table[..6]);

    // Gradient of sum(): each used position row accumulates once per batch
    // element; unused rows get zero.
    let x = Tensor::zeros(&[2, 2, 3]).requires_grad_(true).unwrap();
    pos.forward(&x).unwrap().sum().backward();
    let g = w.grad().unwrap().to_vec();
    assert_eq!(&g[..6], &[2.0; 6], "used rows: one per batch element");
    assert!(g[6..].iter().all(|&v| v == 0.0), "unused rows stay zero");

    // Too-long sequences are refused.
    assert!(pos.forward(&Tensor::zeros(&[1, 9, 3])).is_err());
}

#[test]
fn positions_module_composes_with_params() {
    // The table is a real Param: an optimizer can train it.
    let rng = Rng::new(22);
    let pos = LearnedPositionalEmbedding::new(4, 2, &rng);
    let params: Vec<Param> = pos.named_parameters().into_iter().map(|(_, p)| p).collect();
    let mut opt = ferro_core::optim::Sgd::new(params.clone(), 0.1);
    let before = params[0].tensor().to_vec();
    let x = Tensor::zeros(&[1, 4, 2]).requires_grad_(true).unwrap();
    pos.forward(&x).unwrap().sum().backward();
    opt.step();
    let after = params[0].tensor().to_vec();
    assert!(before.iter().zip(&after).any(|(a, b)| a != b));
}
