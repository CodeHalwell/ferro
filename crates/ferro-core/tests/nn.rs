use ferro_core::nn::{Linear, Module, Relu, Sequential};
use ferro_core::{Rng, Tensor};

fn mlp(rng: &Rng) -> Sequential {
    Sequential::new(vec![
        Box::new(Linear::new(3, 4, rng)),
        Box::new(Relu),
        Box::new(Linear::new(4, 1, rng)),
    ])
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
