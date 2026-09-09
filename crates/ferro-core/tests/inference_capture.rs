use ferro_core::{capture, Tensor};

#[test]
fn capture_tagged_activations() {
    let x = Tensor::from_vec(vec![1., 2.], &[2]).unwrap();
    let outputs = capture(|| vec![x.gelu(), x.gelu_erf(), x.silu(), x.tanh(), x.sqrt()]);
    for y in outputs {
        assert_eq!(Graph::from_root(&y).nodes.len(), 2);
        assert!(Graph::from_root(&y).nodes[&y.id()].tag.is_some());
    }
}

#[test]
fn capture_does_not_interfere_with_mixed_backward() {
    let x = Tensor::scalar(2.);
    let w = Tensor::scalar(3.).requires_grad_(true).unwrap();
    let y = capture(|| x.exp().mul(&w).unwrap());
    y.backward();
    assert_eq!(w.grad().unwrap().to_vec(), vec![2.0f32.exp()]);
    assert!(x.grad().is_none());
}

#[test]
fn capture_nesting_unwind_and_thread_isolation() {
    capture(|| {
        let y = capture(|| Tensor::scalar(1.).exp());
        assert_eq!(Graph::from_root(&y.neg()).nodes.len(), 3);
        let caught = std::panic::catch_unwind(|| capture(|| panic!("test")));
        assert!(caught.is_err());
        assert_eq!(Graph::from_root(&Tensor::scalar(1.).exp()).nodes.len(), 2);
        std::thread::spawn(|| {
            assert_eq!(Graph::from_root(&Tensor::scalar(1.).exp()).nodes.len(), 1);
        })
        .join()
        .unwrap();
    });
    assert!(std::panic::catch_unwind(|| capture(|| panic!("outer"))).is_err());
    assert_eq!(Graph::from_root(&Tensor::scalar(1.).exp()).nodes.len(), 1);
}

#[test]
fn capture_discards_backward_closure() {
    let token = std::sync::Arc::new(());
    let saved = token.clone();
    let y = capture(|| {
        Tensor::scalar(1.).record_fn(vec![Tensor::scalar(1.)], move |_| {
            let _keep = &saved;
            vec![Tensor::scalar(1.)]
        })
    });
    assert_eq!(std::sync::Arc::strong_count(&token), 1);
    assert_eq!(Graph::from_root(&y).nodes.len(), 2);
}
use ferro_core::graph::Graph;

#[test]
fn capture_keeps_untagged_barriers_visible() {
    let x = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
    let outputs = capture(|| {
        vec![
            x.cumsum(1).unwrap(),
            x.log_softmax(1).unwrap(),
            x.logsumexp(1).unwrap(),
            x.prod_dim(1, true).unwrap(),
            x.pad_constant(&[0, 0, 1, 1], 0.).unwrap(),
            x.topk(1, 1).unwrap().0,
        ]
    });
    for y in outputs {
        assert!(
            Graph::from_root(&y).nodes.len() > 1,
            "barrier disappeared: {:?}",
            y.shape()
        );
        assert!(ferro_core::graph::CompiledChain::compile(&capture(|| y.neg())).is_err());
    }
}

#[test]
fn capture_keeps_indexing_normalization_and_pool_barriers() {
    let x = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
    let w = Tensor::ones(&[2]);
    let b = Tensor::zeros(&[2]);
    let index = Tensor::from_vec_i64(vec![0, 1, 1, 0], &[2, 2]).unwrap();
    let outputs = capture(|| {
        vec![
            x.gather(1, &index).unwrap(),
            x.scatter(1, &index, &x).unwrap(),
            x.scatter_add(1, &index, &x).unwrap(),
            x.rope_cached(10000.).unwrap(),
            x.group_norm(1, &w, &b, 1e-5).unwrap(),
            x.batch_norm(&w, &b, &b, &w, 1e-5, false, 0.1)
                .unwrap()
                .output,
            x.bias_add_activation(&b, ferro_core::fused_ops::Act::Relu)
                .unwrap(),
            Tensor::ones(&[1, 1, 2, 2]).avg_pool2d(2, 2).unwrap(),
        ]
    });
    for y in outputs {
        assert!(
            Graph::from_root(&y).nodes.len() > 1,
            "barrier disappeared: {:?}",
            y.shape()
        );
        assert!(ferro_core::graph::CompiledChain::compile(&capture(|| y.neg())).is_err());
    }
}

#[test]
fn captured_chain_replays_current_leaves() {
    let x = Tensor::from_vec(vec![1., 2.], &[2]).unwrap();
    let y = capture(|| x.exp().neg());
    let compiled = ferro_core::graph::CompiledChain::compile(&y).unwrap();
    assert_eq!(compiled.num_steps(), 2);
    assert_eq!(compiled.replay().unwrap().to_vec(), y.to_vec());
    x.fill_(3.).unwrap();
    let replay = capture(|| compiled.replay().unwrap());
    assert_eq!(replay.to_vec(), x.exp().neg().to_vec());
    assert!(!replay.requires_grad());
    assert_eq!(Graph::from_root(&replay).nodes.len(), 1);
}

#[test]
fn capture_records_without_enabling_gradients() {
    let x = Tensor::from_vec(vec![1., 2.], &[2]).unwrap();
    assert_eq!(Graph::from_root(&x.exp()).nodes.len(), 1);
    let y = capture(|| x.exp().neg());
    let graph = Graph::from_root(&y);
    assert_eq!(graph.nodes.len(), 3);
    assert!(graph.nodes[&y.id()].tag.is_some());
    assert!(!y.requires_grad());
    assert!(!x.requires_grad());
    assert_eq!(Graph::from_root(&x.exp()).nodes.len(), 1);
}
