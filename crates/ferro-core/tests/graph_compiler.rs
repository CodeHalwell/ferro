use ferro_core::graph::{Graph, NodeKind};

fn rg(v: Vec<f32>, shape: &[usize]) -> ferro_core::Tensor {
    ferro_core::Tensor::from_vec(v, shape)
        .unwrap()
        .requires_grad_(true)
        .unwrap()
}

#[test]
fn detects_gelu_bias_chain() {
    // x @ w -> gelu -> + bias -> sum: one matmul barrier then a 2-link
    // pointwise chain (gelu unary, bias-add binary).
    let x = rg(vec![0.1; 8], &[4, 2]);
    let w = rg(vec![0.5; 8], &[2, 4]);
    let bias = rg(vec![-0.3; 4], &[4]);
    let loss = Graph::capture(|| {
        let h = x.matmul(&w).unwrap();
        h.gelu().add(&bias).unwrap().sum()
    });
    let plan = loss.plan_fusion();

    assert_eq!(plan.chains.len(), 1);
    assert_eq!(plan.chains[0].nodes.len(), 2);
    assert_eq!(plan.launches_saved(), 1);
    assert_eq!(plan.launches_before, 4); // matmul, gelu, add, sum
    assert_eq!(plan.launches_after, 3);

    // Chain is gelu-out followed by add-out; the matmul and the sum are
    // barriers and appear in no chain.
    let chain: Vec<NodeKind> = plan.chains[0]
        .nodes
        .iter()
        .map(|&id| loss.nodes[&id].kind)
        .collect();
    assert_eq!(chain, vec![NodeKind::Unary, NodeKind::Binary]);
    for &id in &plan.chains[0].nodes {
        assert_ne!(loss.nodes[&id].kind, NodeKind::MatMul);
    }
}

#[test]
fn matmul_is_a_fusion_barrier() {
    // gelu -> + c -> @ d -> silu -> sum: chain of 2 before the matmul, then
    // silu alone (single node, nothing to fuse).
    let a = rg(vec![0.2; 6], &[2, 3]);
    let c = rg(vec![0.1; 6], &[2, 3]);
    let d = rg(vec![0.4; 9], &[3, 3]);
    let g = Graph::capture(|| {
        let t = a.gelu().add(&c).unwrap();
        t.matmul(&d).unwrap().silu().sum()
    });
    let plan = g.plan_fusion();

    assert_eq!(plan.chains.len(), 1);
    assert_eq!(plan.chains[0].nodes.len(), 2);
    assert_eq!(plan.launches_saved(), 1);
    let mm = g
        .nodes
        .values()
        .filter(|n| n.kind == NodeKind::MatMul)
        .count();
    assert_eq!(mm, 1);
    assert!(!plan.chains[0].nodes.contains(
        &g.nodes
            .values()
            .find(|n| n.kind == NodeKind::MatMul)
            .unwrap()
            .id
    ));
}

#[test]
fn longer_chain_and_fanout() {
    // relu -> gelu -> * scale: a 3-link chain saves 2 launches.
    let x = rg(vec![0.5; 4], &[4]);
    let s = rg(vec![2.0; 4], &[4]);
    let g = Graph::capture(|| x.relu().gelu().mul(&s).unwrap().sum());
    let plan = g.plan_fusion();
    assert_eq!(plan.chains.len(), 1);
    assert_eq!(plan.chains[0].nodes.len(), 3);
    assert_eq!(plan.launches_saved(), 2);

    // Fan-out blocks fusion THROUGH y: relu -> gelu must not fuse because y
    // feeds two continuations. The downstream gelu -> mul pair still fuses.
    let b = rg(vec![1.0; 4], &[4]);
    let (y, loss2) = {
        let y = x.relu();
        let z = y.add(&b).unwrap();
        let w = y.gelu().mul(&z).unwrap();
        (y, w.sum())
    };
    let g2 = Graph::from_root(&loss2);
    let plan2 = g2.plan_fusion();
    assert!(
        plan2.chains.iter().all(|c| !c.nodes.contains(&y.id())),
        "fan-out node must not appear in any chain"
    );
    assert_eq!(plan2.launches_saved(), 1);

    // A side input that is itself an intermediate does not break the
    // downstream chain (add -> mul stays fusible), but the fanned-out mul
    // itself cannot join any chain.
    let (y3, loss3) = {
        let y = x.mul(&x).unwrap();
        (y.clone(), y.add(&b).unwrap().mul(&y).unwrap().sum())
    };
    let g3 = Graph::from_root(&loss3);
    let plan3 = g3.plan_fusion();
    assert_eq!(plan3.chains.len(), 1);
    assert_eq!(plan3.chains[0].nodes.len(), 2);
    assert!(plan3.chains[0].nodes.iter().all(|&n| n != y3.id()));
    assert_eq!(plan3.launches_saved(), 1);
}
