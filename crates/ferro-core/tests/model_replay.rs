use ferro_core::{capture, Tensor};
use ferro_core::graph::CompiledChain;

#[test]
fn attention_layout_norm_replays_current_leaves() {
    let x = Tensor::from_vec((0..24).map(|i| i as f32 * 0.03 - 0.2).collect(), &[3, 8]).unwrap();
    let w = Tensor::full(&[8], 1.1);
    let b = Tensor::full(&[8], 0.05);
    let forward = || {
        let z = x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap();
        let q = z.reshape(&[3, 2, 4]).unwrap().transpose(0, 1).unwrap();
        let scores = q.bmm(&q.transpose(1, 2).unwrap()).unwrap();
        let a = scores.softmax(2).unwrap().bmm(&q).unwrap();
        x.add(&a.transpose(0, 1).unwrap().reshape(&[3, 8]).unwrap()).unwrap()
    };
    let root = capture(forward);
    let compiled = CompiledChain::compile(&root).unwrap();
    assert_eq!(compiled.num_steps(), 10);
    x.copy_from(&Tensor::from_vec((0..24).map(|i| (i as f32 * 0.7).sin()).collect(), &[3, 8]).unwrap()).unwrap();
    let actual = compiled.replay().unwrap();
    for (a, b) in actual.to_vec().iter().zip(forward().to_vec()) { assert!((a-b).abs() < 1e-6); }
    assert_ne!(actual.to_vec(), root.to_vec());
}

#[test]
fn residual_mlp_replays_current_input_and_weights() {
    let x = Tensor::from_vec(vec![0.2, -0.4, 0.7, 0.9], &[2, 2]).unwrap();
    let w = Tensor::from_vec(vec![0.3, -0.2, 0.5, 0.8], &[2, 2]).unwrap();
    let b = Tensor::full(&[2], 0.1);
    let forward = || x.add(&x.matmul(&w).unwrap().add(&b).unwrap().gelu().matmul(&w).unwrap()).unwrap();
    let root = capture(forward);
    let compiled = CompiledChain::compile(&root).unwrap();
    assert_eq!(compiled.num_steps(), 5);
    let matmul = CompiledChain::compile(&capture(|| x.matmul(&w).unwrap())).unwrap();
    let replay_in_capture = capture(|| matmul.replay().unwrap());
    assert_eq!(ferro_core::graph::Graph::from_root(&replay_in_capture).nodes.len(), 1);
    assert_eq!(compiled.replay().unwrap().to_vec(), forward().to_vec());
    x.copy_from(&Tensor::full(&[2, 2], 0.6)).unwrap();
    w.copy_from(&Tensor::full(&[2, 2], 0.2)).unwrap();
    let got = compiled.replay().unwrap();
    assert!(!got.requires_grad());
    assert_eq!(got.to_vec(), forward().to_vec());
    assert_ne!(got.to_vec(), root.to_vec());
}
