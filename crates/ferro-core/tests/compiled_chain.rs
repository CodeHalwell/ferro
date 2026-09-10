use ferro_core::graph::CompiledChain;
use ferro_core::Tensor;

fn leaf(v: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(v.to_vec(), shape).unwrap().requires_grad_(true).unwrap()
}

#[test]
fn computed_side_branch_reads_current_leaves() {
    let px = ferro_core::Param::new(leaf(&[2., -3., 4.], &[3]));
    let py = ferro_core::Param::new(leaf(&[5., -2., -1.], &[3]));
    let (x, y) = (px.tensor(), py.tensor());
    let build = || x.relu().mul(&y.relu()).unwrap().relu();
    let root = build();
    let h = CompiledChain::compile(&root).unwrap();
    assert_eq!(h.num_steps(), 4);
    assert_eq!(h.num_operands(), 2);
    assert_eq!(h.replay().unwrap().to_vec(), root.to_vec());
    let mut opt = ferro_core::optim::Sgd::new(vec![px, py], 0.5);
    x.sum().add(&y.sum()).unwrap().backward();
    opt.step();
    assert_ne!(h.replay().unwrap().to_vec(), root.to_vec());
    assert_eq!(h.replay().unwrap().to_vec(), build().to_vec());
    assert_eq!(ferro_core::graph::Graph::from_root(&root).eval_fused().unwrap().to_vec(), build().to_vec());
}

#[test]
fn expanding_seed_is_replayed() {
    let x = leaf(&[2., -3., 4.], &[3]);
    let y = leaf(&[1.; 6], &[2, 3]);
    let root = x.relu().mul(&y).unwrap().relu();
    let h = CompiledChain::compile(&root).unwrap();
    assert_eq!(h.replay().unwrap().shape(), &[2, 3]);
    assert_eq!(h.replay().unwrap().to_vec(), root.to_vec());
    let g = ferro_core::graph::Graph::from_root(&root);
    assert!(g.plan_fusion().chains[0].resolve(&g).is_err());
}

#[test]
fn duplicate_seed_operands_work() {
    let x = leaf(&[2., -3., 4.], &[3]);
    let root = x.mul(&x).unwrap().sub(&x).unwrap();
    let h = CompiledChain::compile(&root).unwrap();
    assert_eq!(h.num_operands(), 1);
    assert_eq!(h.replay().unwrap().to_vec(), root.to_vec());
}

#[test]
fn legacy_binary_seed_is_already_computed() {
    let x = leaf(&[2., -3., 4.], &[3]);
    let y = leaf(&[5., 2., -1.], &[3]);
    let root = x.mul(&y).unwrap().sub(&y).unwrap();
    let g = ferro_core::graph::Graph::from_root(&root);
    let plan = g.plan_fusion();
    let exec = plan.chains[0].resolve(&g).unwrap();
    assert_eq!(exec.steps.len(), 1);
    assert_eq!(plan.chains[0].run(&exec).unwrap().to_vec(), root.to_vec());
}

#[test]
fn first_unary_and_current_upstream_leaves_are_replayed() {
    let px = ferro_core::Param::new(leaf(&[-2., 3., 4.], &[3]));
    let py = ferro_core::Param::new(leaf(&[2., 4., 8.], &[3]));
    let pz = ferro_core::Param::new(leaf(&[1., 2., 3.], &[3]));
    let (x, y, z) = (px.tensor(), py.tensor(), pz.tensor());
    let build = || x.relu().sub(&y).unwrap().div(&z).unwrap();
    let root = build();
    let h = CompiledChain::compile(&root).unwrap();
    assert_eq!(h.num_steps(), 3);
    assert_eq!(h.replay().unwrap().to_vec(), root.to_vec());
    let mut opt = ferro_core::optim::Sgd::new(vec![px, py, pz], 0.5);
    x.sum().add(&y.sum()).unwrap().add(&z.sum()).unwrap().backward();
    opt.step();
    let got = h.replay().unwrap();
    assert_ne!(got.to_vec(), root.to_vec());
    assert_eq!(got.to_vec(), build().to_vec());
    assert!(!got.requires_grad());
}

#[test]
fn broadcast_right_leaf_is_supported() {
    let x = leaf(&[-2., 3., 4., 5., -6., 7.], &[2, 3]);
    let y = leaf(&[2., 4., 8.], &[3]);
    let root = x.relu().sub(&y).unwrap().div(&y).unwrap();
    let h = CompiledChain::compile(&root).unwrap();
    assert_eq!(h.num_operands(), 2);
    assert_eq!(h.replay().unwrap().shape(), &[2, 3]);
    assert_eq!(h.replay().unwrap().to_vec(), root.to_vec());
}

#[test]
fn unsupported_graphs_are_rejected_not_frozen() {
    let x = leaf(&[2., 3., 4.], &[3]);
    let y = leaf(&[5., 2., 1.], &[3]);
    let u = x.relu();
    for root in [
        x.sub(&y.relu()).unwrap().relu(),
        x.div(&y.relu()).unwrap().relu(),
        u.mul(&u).unwrap(),
        u.sub(&x).unwrap().div(&u.add(&y).unwrap()).unwrap(),
    ] {
        let h = CompiledChain::compile(&root).unwrap();
        assert_eq!(h.replay().unwrap().to_vec(), root.to_vec());
    }
    assert!(CompiledChain::compile(&x.sum().relu()).is_err());
    assert!(CompiledChain::compile(&x).is_err());
}

#[test]
fn oversized_broadcast_metadata_is_rejected_without_truncation() {
    let x = leaf(&[], &[0, u32::MAX as usize + 1]);
    let y = leaf(&[1.], &[1]);
    let root = x.add(&y).unwrap();
    assert!(CompiledChain::compile(&root).is_err());
}

#[test]
fn scalar_broadcasts_replay_in_both_operand_positions() {
    let x = leaf(&[2., 3., 4.], &[3]);
    let s = leaf(&[2.], &[]);
    for root in [x.sub(&s).unwrap(), s.div(&x).unwrap().relu()] {
        let h = CompiledChain::compile(&root).unwrap();
        assert_eq!(h.replay().unwrap().to_vec(), root.to_vec());
    }
}

#[test]
fn binary_first_replays_all_steps() {
    let x = leaf(&[2., -3., 4.], &[3]);
    let y = leaf(&[5., 2., -1.], &[3]);
    let z = leaf(&[1., 4., 2.], &[3]);
    let root = x.mul(&y).unwrap().add(&z).unwrap();
    let h = CompiledChain::compile(&root).unwrap();
    assert_eq!(h.num_steps(), 2);
    assert_eq!(h.replay().unwrap().to_vec(), root.to_vec());
}
