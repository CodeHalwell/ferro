use ferro_core::nn::{cross_entropy, cross_entropy_indices, one_hot};
use ferro_core::nn::{scaled_dot_product_attention, Embedding, Gelu, LayerNorm, Linear, Module, Relu, RmsNorm, Sequential};
use ferro_core::testkit::grad_check;
use ferro_core::{Rng, Tensor};

fn mlp(rng: &Rng) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(3, 4, rng)),
        Box::new(Relu),
        Box::new(Linear::new(4, 1, rng)),
    ])
}

#[test]
fn cross_entropy_value_and_grad() {
    // Uniform logits over 2 classes: loss is exactly ln(2).
    let logits = Tensor::zeros(&[2, 2]);
    let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
    let loss = cross_entropy(&logits, &targets).unwrap();
    assert!((loss.item() - 2f32.ln()).abs() < 1e-5, "expected ln2, got {}", loss.item());

    let logits = Tensor::from_vec(vec![0.5, -0.3, 1.2, 0.1, -0.8, 0.4], &[2, 3]).unwrap();
    let t = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0], &[2, 3]).unwrap();
    grad_check(&[logits], |l| cross_entropy(&l[0], &t).unwrap());
}

#[test]
fn cross_entropy_training_separates_classes() {
    // A linear classifier driven by cross_entropy must fit a separable toy set.
    let rng = Rng::new(7);
    let x = Tensor::from_vec(vec![2.0, 0.1, 1.8, -0.2, -2.1, 0.0, -1.7, 0.3], &[4, 2]).unwrap();
    let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], &[4, 2]).unwrap();
    let layer = Linear::new(2, 2, &rng);
    let mut opt = ferro_core::optim::Sgd::new(layer.parameters(), 0.5);
    let mut first = 0.0;
    let mut last = 0.0;
    for step in 0..100 {
        let loss = cross_entropy(&layer.forward(&x).unwrap(), &targets).unwrap();
        opt.zero_grad();
        loss.backward();
        opt.step();
        if step == 0 {
            first = loss.item();
        }
        last = loss.item();
    }
    assert!(last < 0.05 && last < first * 0.2, "loss did not converge: {first} -> {last}");
}

#[test]
fn one_hot_values_and_errors() {
    let ids = Tensor::from_vec_i64(vec![2, 0], &[2]).unwrap();
    let oh = one_hot(&ids, 3).unwrap();
    assert_eq!(oh.shape(), &[2, 3]);
    assert_eq!(oh.to_vec(), vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);

    assert!(one_hot(&ids, 2).is_err());
    assert!(one_hot(&Tensor::from_vec_i64(vec![-1], &[1]).unwrap(), 3).is_err());
    assert!(one_hot(&Tensor::from_vec(vec![1.0], &[1]).unwrap(), 3).is_err());
}

#[test]
fn cross_entropy_indices_matches_one_hot() {
    let logits = Tensor::from_vec(vec![0.5, -0.3, 1.2, 0.1, -0.8, 0.4], &[2, 3]).unwrap();
    let ids = Tensor::from_vec_i64(vec![2, 0], &[2]).unwrap();

    let via_ids = cross_entropy_indices(&logits, &ids).unwrap();
    let via_one_hot = cross_entropy(&logits, &one_hot(&ids, 3).unwrap()).unwrap();
    assert!((via_ids.item() - via_one_hot.item()).abs() < 1e-6);

    grad_check(&[logits], |l| cross_entropy_indices(&l[0], &ids).unwrap());
}

#[test]
fn linear_forward_shape_and_params() {
    let rng = Rng::new(0);
    let layer = Linear::new(3, 5, &rng);
    let x = Tensor::from_vec((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
    let y = layer.forward(&x).unwrap();
    assert_eq!(y.shape(), &[2, 5]);
    assert_eq!(layer.parameters().len(), 2);
}

#[test]
fn mlp_backward_populates_grads() {
    let rng = Rng::new(1);
    let model = mlp(&rng);
    let x = Tensor::from_vec((0..12).map(|v| v as f32 * 0.1).collect(), &[4, 3]).unwrap();
    let out = model.forward(&x).unwrap();
    assert_eq!(out.shape(), &[4, 1]);
    let loss = out.mean();
    loss.backward();
    for p in model.parameters() {
        let g = p.grad().expect("parameter should have grad");
        assert_eq!(g.shape(), p.tensor().shape());
    }
}

#[test]
fn training_loop_decreases_loss() {
    let rng = Rng::new(7);
    let model = mlp(&rng);

    // Regress y = sum of inputs.
    let inputs: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.5, 0.1, 0.4, 0.9, 0.2, 0.1, 0.3, 0.3, 0.3, 0.0, 0.7, 0.2, 0.6, 0.1, 0.1,
    ];
    let batch = inputs.len() / 3;
    let x = Tensor::from_vec(inputs.clone(), &[batch, 3]).unwrap();
    let targets: Vec<f32> = inputs.chunks(3).map(|c| c.iter().sum()).collect();
    let y = Tensor::from_vec(targets, &[batch, 1]).unwrap();

    let lr = 0.1;
    let mut first_loss = f32::NAN;
    let mut last_loss = f32::NAN;
    for step in 0..200 {
        let pred = model.forward(&x).unwrap();
        let diff = pred.sub(&y).unwrap();
        let loss = diff.mul(&diff).unwrap().mean();
        let loss_val = loss.item();
        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        loss.backward();
        for p in model.parameters() {
            let grad = p.grad().unwrap().to_vec();
            let cur = p.tensor();
            let shape = cur.shape().to_vec();
            let updated: Vec<f32> =
                cur.to_vec().iter().zip(grad.iter()).map(|(w, g)| w - lr * g).collect();
            p.set(Tensor::from_vec(updated, &shape).unwrap());
            p.zero_grad();
        }
    }

    assert!(last_loss < first_loss * 0.5, "loss did not decrease: {first_loss} -> {last_loss}");
}

#[test]
fn layernorm_normalizes() {
    let ln = LayerNorm::new(4);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, -0.5, 0.7, 2.3, -1.1], &[2, 4]).unwrap();
    let out = ln.forward(&x).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    for row in out.to_vec().chunks(4) {
        let mean = row.iter().sum::<f32>() / 4.0;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "row mean not ~0: {mean}");
        assert!((var - 1.0).abs() < 1e-3, "row var not ~1: {var}");
    }
}

#[test]
fn layernorm_grad() {
    let ln = LayerNorm::new(3);
    let x = Tensor::from_vec(vec![0.5, -0.3, 1.2, 0.1, -0.8, 0.4], &[2, 3]).unwrap();
    let w = Tensor::from_vec(vec![0.7, -1.3, 0.4, 2.1, -0.6, 0.9], &[2, 3]).unwrap();
    grad_check(&[x], |t| ln.forward(&t[0]).unwrap().mul(&w).unwrap().sum());
}

#[test]
fn layernorm_in_mlp_trains() {
    let rng = Rng::new(7);
    let model = Sequential::new(vec![
        Box::new(Linear::new(3, 4, &rng)),
        Box::new(LayerNorm::new(4)),
        Box::new(Relu),
        Box::new(Linear::new(4, 1, &rng)),
    ]);

    // Regress y = sum of inputs.
    let inputs: Vec<f32> = vec![
        0.1, 0.2, 0.3, 0.5, 0.1, 0.4, 0.9, 0.2, 0.1, 0.3, 0.3, 0.3, 0.0, 0.7, 0.2, 0.6, 0.1, 0.1,
    ];
    let batch = inputs.len() / 3;
    let x = Tensor::from_vec(inputs.clone(), &[batch, 3]).unwrap();
    let targets: Vec<f32> = inputs.chunks(3).map(|c| c.iter().sum()).collect();
    let y = Tensor::from_vec(targets, &[batch, 1]).unwrap();

    let mut opt = ferro_core::optim::Sgd::new(model.parameters(), 0.1);
    let mut first_loss = f32::NAN;
    let mut last_loss = f32::NAN;
    for step in 0..200 {
        let pred = model.forward(&x).unwrap();
        let diff = pred.sub(&y).unwrap();
        let loss = diff.mul(&diff).unwrap().mean();
        if step == 0 {
            first_loss = loss.item();
        }
        last_loss = loss.item();
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    assert!(last_loss < first_loss * 0.5, "loss did not decrease: {first_loss} -> {last_loss}");
}

#[test]
fn rmsnorm_normalizes_any_rank() {
    let rn = RmsNorm::new(4);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, -0.5, 0.7, 2.3, -1.1, 0.2, 0.9, -0.4, 1.6], &[1, 3, 4]).unwrap();
    let out = rn.forward(&x).unwrap();
    assert_eq!(out.shape(), &[1, 3, 4]);
    for row in out.to_vec().chunks(4) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((ms - 1.0).abs() < 1e-3, "row mean-square not ~1: {ms}");
    }
}

#[test]
fn rmsnorm_grad() {
    let rn = RmsNorm::new(3);
    let x = Tensor::from_vec(vec![0.5, -0.3, 1.2, 0.1, -0.8, 0.4], &[2, 3]).unwrap();
    let w = Tensor::from_vec(vec![0.7, -1.3, 0.4, 2.1, -0.6, 0.9], &[2, 3]).unwrap();
    grad_check(&[x], |t| rn.forward(&t[0]).unwrap().mul(&w).unwrap().sum());
}

#[test]
fn embedding_module_shapes_and_grad() {
    let rng = Rng::new(5);
    let emb = Embedding::new(6, 3, &rng);
    let ids = Tensor::from_vec_i64(vec![1, 4, 1, 0], &[2, 2]).unwrap();
    let out = emb.forward(&ids).unwrap();
    assert_eq!(out.shape(), &[2, 2, 3]);

    // Rows for the same id must match, and the duplicated id's grad must
    // accumulate both contributions in the weight.
    let o = out.to_vec();
    assert_eq!(&o[0..3], &o[6..9]);
    out.sum().backward();
    let g = emb.parameters()[0].grad().unwrap().to_vec();
    for (row, want) in [(0, 1.0), (1, 2.0), (4, 1.0), (2, 0.0)] {
        for j in 0..3 {
            assert_eq!(g[row * 3 + j], want, "weight row {row}");
        }
    }
}

#[test]
fn gelu_module_matches_op() {
    let x = Tensor::from_vec(vec![-1.0, 0.5, 2.0], &[3]).unwrap();
    assert_eq!(Gelu.forward(&x).unwrap().to_vec(), x.gelu().to_vec());
}

#[test]
fn attention_uniform_when_query_is_zero() {
    // q = 0 makes every score 0, so softmax is uniform and each output row is
    // the mean of the attendable v rows: all of v without the causal mask,
    // the prefix v[0..=i] with it.
    let q = Tensor::zeros(&[1, 3, 2]);
    let k = Tensor::from_vec(vec![0.3, -0.9, 1.2, 0.4, -0.5, 0.8], &[1, 3, 2]).unwrap();
    let v = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]).unwrap();

    let full = scaled_dot_product_attention(&q, &k, &v, false).unwrap().to_vec();
    for row in full.chunks(2) {
        assert!((row[0] - 3.0).abs() < 1e-5 && (row[1] - 4.0).abs() < 1e-5);
    }

    let causal = scaled_dot_product_attention(&q, &k, &v, true).unwrap().to_vec();
    let want = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
    for (g, w) in causal.iter().zip(want) {
        assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
    }
}

#[test]
fn attention_rejects_mismatched_shapes() {
    let q = Tensor::zeros(&[1, 2, 4]);
    let k = Tensor::zeros(&[1, 3, 4]);
    let v = Tensor::zeros(&[1, 3, 2]);
    assert!(scaled_dot_product_attention(&q, &k, &v, false).is_ok());
    assert!(scaled_dot_product_attention(&q, &Tensor::zeros(&[1, 3, 2]), &v, false).is_err());
    assert!(scaled_dot_product_attention(&q, &k, &Tensor::zeros(&[1, 2, 2]), false).is_err());
    assert!(scaled_dot_product_attention(&Tensor::zeros(&[2, 4]), &k, &v, false).is_err());
}

#[test]
fn attention_grad() {
    let q = Tensor::from_vec(vec![0.4, -0.7, 1.1, 0.2, -0.3, 0.9, 0.6, -1.2], &[1, 2, 4]).unwrap();
    let k = Tensor::from_vec((0..12).map(|i| ((i * 5 % 7) as f32 - 3.0) / 3.0).collect(), &[1, 3, 4]).unwrap();
    let v = Tensor::from_vec((0..6).map(|i| i as f32 / 3.0 - 1.0).collect(), &[1, 3, 2]).unwrap();
    grad_check(&[q, k, v], |t| scaled_dot_product_attention(&t[0], &t[1], &t[2], true).unwrap().sum());
}

#[test]
fn transformer_primitives_compose_end_to_end() {
    // Embedding -> RmsNorm -> RoPE'd self-attention -> Gelu, and gradients
    // reach the embedding table: the op set milestone M3 needs, in one graph.
    let rng = Rng::new(11);
    let emb = Embedding::new(10, 4, &rng);
    let rn = RmsNorm::new(4);
    let ids = Tensor::from_vec_i64(vec![3, 7, 1], &[1, 3]).unwrap();
    let pos = Tensor::from_vec_i64(vec![0, 1, 2], &[3]).unwrap();

    let h = rn.forward(&emb.forward(&ids).unwrap()).unwrap();
    let q = h.rope(&pos, 10000.0).unwrap();
    let k = h.rope(&pos, 10000.0).unwrap();
    let out = scaled_dot_product_attention(&q, &k, &h, true).unwrap().gelu();
    assert_eq!(out.shape(), &[1, 3, 4]);

    out.sum().backward();
    let g = emb.parameters()[0].grad().expect("embedding weight grad");
    assert_eq!(g.shape(), &[10, 4]);
    let gsum: f32 = g.to_vec().iter().map(|x| x.abs()).sum();
    assert!(gsum > 0.0, "gradient did not reach the embedding table");
}
