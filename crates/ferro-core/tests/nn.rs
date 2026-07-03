use ferro_core::nn::{cross_entropy, cross_entropy_indices, one_hot};
use ferro_core::nn::{LayerNorm, Linear, Module, Relu, Sequential};
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
