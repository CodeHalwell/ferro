use ferro_core::gnn::{scatter, Aggr, EdgeIndex, GcnConv, GraphModule, MessagePassing};
use ferro_core::nn::cross_entropy_indices;
use ferro_core::optim::Adam;
use ferro_core::testkit::grad_check;
use ferro_core::{Rng, Tensor};

fn t(v: &[f32], shape: &[usize]) -> Tensor { Tensor::from_vec(v.to_vec(), shape).unwrap() }

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() { assert!((x - y).abs() < 1e-5, "elem {i}: {x} vs {y}"); }
}

// Five nodes: a duplicated 1->2 edge, a one-way 0->2 edge, a repeated self
// loop on 3, and node 4 isolated.
fn toy() -> EdgeIndex {
    EdgeIndex::new(5, vec![0, 1, 1, 2, 1, 3, 3, 0], vec![1, 0, 2, 1, 2, 3, 3, 2]).unwrap()
}

// Independent dense D^-1/2 (A + I) D^-1/2 with A[dst][src] counting duplicates.
fn dense_gcn_adjacency(e: &EdgeIndex) -> Tensor {
    let n = e.num_nodes();
    let mut a = vec![0f32; n * n];
    for (&s, &d) in e.src().iter().zip(e.dst()) { if s != d { a[d * n + s] += 1.; } }
    for i in 0..n { a[i * n + i] = 1.; }
    let deg: Vec<f32> = (0..n).map(|i| a[i * n..(i + 1) * n].iter().sum()).collect();
    for d in 0..n { for s in 0..n { a[d * n + s] /= (deg[d] * deg[s]).sqrt(); } }
    t(&a, &[n, n])
}

fn features(n: usize, f: usize) -> Vec<f32> { (0..n * f).map(|i| ((i * 7 % 11) as f32 - 5.) / 5.).collect() }

#[test]
fn scatter_reductions_zero_fill_and_gradients() {
    let x = t(&[1., -2., 3., 4., 5., -6.], &[3, 2]);
    let ids = [2, 0, 2];
    assert_eq!(scatter(&x, &ids, 4, Aggr::Sum).unwrap().to_vec(), vec![3., 4., 0., 0., 6., -8., 0., 0.]);
    assert_eq!(scatter(&x, &ids, 4, Aggr::Mean).unwrap().to_vec(), vec![3., 4., 0., 0., 3., -4., 0., 0.]);
    assert_eq!(scatter(&x, &ids, 4, Aggr::Max).unwrap().to_vec(), vec![3., 4., 0., 0., 5., -2., 0., 0.]);
    assert_eq!(scatter(&x, &ids, 4, Aggr::Min).unwrap().to_vec(), vec![3., 4., 0., 0., 1., -6., 0., 0.]);
    for reduce in [Aggr::Sum, Aggr::Mean, Aggr::Max, Aggr::Min] {
        grad_check(&[t(&[0.3, -0.2, 0.7, 0.4, -0.6, 0.9], &[3, 2])], |xs| {
            let y = scatter(&xs[0], &ids, 4, reduce).unwrap();
            y.mul(&y).unwrap().sum()
        });
    }
    let ties = t(&[2., 2., 1.], &[3]).requires_grad_(true).unwrap();
    scatter(&ties, &[0, 0, 0], 2, Aggr::Min).unwrap().sum().backward();
    assert_eq!(ties.grad().unwrap().to_vec(), vec![0., 0., 1.]);
    assert!(scatter(&x, &[0, 5, 1], 3, Aggr::Max).is_err());
    assert!(scatter(&x, &[0, 1], 3, Aggr::Sum).is_err());
}

#[test]
fn edge_index_validation_self_loops_and_round_trip() {
    assert!(EdgeIndex::new(3, vec![0, 1], vec![2]).is_err());
    assert!(EdgeIndex::new(3, vec![0, 3], vec![1, 2]).is_err());
    assert!(EdgeIndex::from_tensor(&Tensor::from_vec_i64(vec![0, -1], &[2, 1]).unwrap(), 2).is_err());
    assert!(EdgeIndex::from_tensor(&Tensor::from_vec_i64(vec![0, 1, 1], &[3, 1]).unwrap(), 2).is_err());
    assert!(EdgeIndex::from_tensor(&t(&[0., 1.], &[2, 1]), 2).is_err());

    let e = toy();
    let looped = e.add_remaining_self_loops();
    assert_eq!(looped.src(), &[0, 1, 1, 2, 1, 0, 0, 1, 2, 3, 4]);
    assert_eq!(looped.dst(), &[1, 0, 2, 1, 2, 2, 0, 1, 2, 3, 4]);
    assert_eq!(looped.in_degree(), vec![2, 3, 4, 1, 1]);
    assert_eq!(EdgeIndex::from_tensor(&e.to_tensor(), 5).unwrap(), e);

    let dir = std::env::temp_dir().join(format!("ferro_gnn_topology_{}.safetensors", std::process::id()));
    ferro_core::save_safetensors(&dir, &[("edge_index", &e.to_tensor())]).unwrap();
    let loaded = ferro_core::load_safetensors(&dir).unwrap();
    std::fs::remove_file(&dir).unwrap();
    let (_, saved) = loaded.iter().find(|(n, _)| n == "edge_index").unwrap();
    assert_eq!(EdgeIndex::from_tensor(saved, 5).unwrap(), e);
}

#[test]
fn gcn_matches_dense_reference_values_and_gradients() {
    let e = toy();
    let gcn = GcnConv::new(3, 4, &Rng::new(7));
    let params = gcn.parameters();
    params[1].set(t(&[0.1, -0.2, 0.3, 0.05], &[4]));
    let (w, b) = (params[0].tensor(), params[1].tensor());
    let probe = t(&features(5, 4), &[5, 4]);

    let x = t(&features(5, 3), &[5, 3]).requires_grad_(true).unwrap();
    let out = gcn.forward(&x, &e).unwrap();
    out.mul(&probe).unwrap().sum().backward();

    let x2 = x.detach_copy().requires_grad_(true).unwrap();
    let w2 = w.detach_copy().requires_grad_(true).unwrap();
    let b2 = b.detach_copy().requires_grad_(true).unwrap();
    let reference = dense_gcn_adjacency(&e).matmul(&x2.matmul(&w2).unwrap()).unwrap().add(&b2).unwrap();
    reference.mul(&probe).unwrap().sum().backward();

    close(&out.to_vec(), &reference.to_vec());
    close(&x.grad().unwrap().to_vec(), &x2.grad().unwrap().to_vec());
    close(&params[0].grad().unwrap().to_vec(), &w2.grad().unwrap().to_vec());
    close(&params[1].grad().unwrap().to_vec(), &b2.grad().unwrap().to_vec());
    // The isolated node sees only its own self loop.
    close(&out.to_vec()[16..], &x.index_select(0, &[4]).unwrap().matmul(&w).unwrap().add(&b).unwrap().to_vec());

    grad_check(&[t(&features(5, 3), &[5, 3])], |xs| {
        let y = gcn.forward(&xs[0], &e).unwrap();
        y.mul(&y).unwrap().sum()
    });
    assert!(gcn.forward(&t(&features(4, 3), &[4, 3]), &e).is_err());
    assert!(gcn.forward(&t(&features(5, 2), &[5, 2]), &e).is_err());
}

#[test]
fn gcn_without_self_loops_zeroes_sourceless_nodes() {
    let e = EdgeIndex::new(3, vec![0, 1], vec![1, 2]).unwrap();
    let gcn = GcnConv::new(2, 2, &Rng::new(1)).with_self_loops(false);
    let out = gcn.forward(&t(&features(3, 2), &[3, 2]), &e).unwrap().to_vec();
    // Node 0 receives nothing; node 2 hears node 1 at weight deg(1)^-1/2 deg(2)^-1/2 = 1.
    close(&out[..2], &[0., 0.]);
    assert!(out[4..].iter().all(|v| v.is_finite()));
}

#[test]
fn gcn_is_node_permutation_equivariant() {
    let e = toy();
    let gcn = GcnConv::new(3, 4, &Rng::new(3));
    let x = features(5, 3);
    let perm = [3, 0, 4, 1, 2]; // new node i is old node perm[i]
    let mut inv = [0; 5];
    for (i, &p) in perm.iter().enumerate() { inv[p] = i; }
    let pe = EdgeIndex::new(5, e.src().iter().map(|&s| inv[s]).collect(), e.dst().iter().map(|&d| inv[d]).collect()).unwrap();
    let px: Vec<f32> = perm.iter().flat_map(|&p| x[p * 3..p * 3 + 3].to_vec()).collect();
    let out = gcn.forward(&t(&x, &[5, 3]), &e).unwrap();
    let pout = gcn.forward(&t(&px, &[5, 3]), &pe).unwrap();
    close(&pout.to_vec(), &out.index_select(0, &perm).unwrap().to_vec());
}

#[test]
fn batched_graphs_match_separate_forwards_and_pool() {
    let a = toy();
    let b = EdgeIndex::new(3, vec![0, 2, 2], vec![1, 1, 0]).unwrap();
    let gcn = GcnConv::new(3, 2, &Rng::new(11));
    let (xa, xb) = (t(&features(5, 3), &[5, 3]), t(&features(3, 3), &[3, 3]).neg());
    let (edges, owner) = EdgeIndex::batch(&[a.clone(), b.clone()]);
    assert_eq!(owner, vec![0, 0, 0, 0, 0, 1, 1, 1]);
    let joint = gcn.forward(&Tensor::cat(&[xa.clone(), xb.clone()], 0).unwrap(), &edges).unwrap();
    let separate = Tensor::cat(&[gcn.forward(&xa, &a).unwrap(), gcn.forward(&xb, &b).unwrap()], 0).unwrap();
    close(&joint.to_vec(), &separate.to_vec());
    let pooled = scatter(&joint, &owner, 2, Aggr::Mean).unwrap().to_vec();
    let sep = separate.to_vec();
    for f in 0..2 {
        let mean_b: f32 = (5..8).map(|n| sep[n * 2 + f]).sum::<f32>() / 3.;
        assert!((pooled[2 + f] - mean_b).abs() < 1e-5);
    }
}

// EdgeConv-style layer: max over neighbours of (x_j - x_i), exercising a
// custom message that needs target rows and a non-sum aggregation.
struct MaxDiff;

impl MessagePassing for MaxDiff {
    fn aggr(&self) -> Aggr { Aggr::Max }
    fn message(&self, x: &Tensor, edges: &EdgeIndex) -> ferro_core::Result<Tensor> {
        x.index_select(0, edges.src())?.sub(&x.index_select(0, edges.dst())?)
    }
    fn update(&self, aggregated: Tensor, x: &Tensor) -> ferro_core::Result<Tensor> { aggregated.add(x) }
}

#[test]
fn custom_message_passing_layer() {
    let e = EdgeIndex::new(4, vec![1, 2, 0, 3], vec![0, 0, 1, 1]).unwrap();
    let x = t(&[0.5, -1., 2., 0.25], &[4, 1]);
    // Node 0: max(-1.5, 1.5) + 0.5; node 1: max(1.5, 1.25) - 1; nodes 2, 3 isolated: 0 + x.
    assert_eq!(MaxDiff.propagate(&x, &e).unwrap().to_vec(), vec![2., 0.5, 2., 0.25]);
    grad_check(&[x.detach_copy()], |xs| {
        let y = MaxDiff.propagate(&xs[0], &e).unwrap();
        y.mul(&y).unwrap().sum()
    });
    assert!(MaxDiff.propagate(&t(&[1., 2.], &[2, 1]), &e).is_err());
}

#[test]
fn two_layer_gcn_separates_two_communities() {
    // Two 4-cliques joined by one bridge edge; label = community.
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for base in [0, 4] {
        for i in 0..4 { for j in 0..4 { if i != j { src.push(base + i); dst.push(base + j); } } }
    }
    src.extend([3, 4]);
    dst.extend([4, 3]);
    let e = EdgeIndex::new(8, src, dst).unwrap();
    let x = Tensor::from_vec((0..64).map(|i| if i % 9 == 0 { 1. } else { 0. }).collect(), &[8, 8]).unwrap();
    let labels = Tensor::from_vec_i64(vec![0, 0, 0, 0, 1, 1, 1, 1], &[8]).unwrap();
    let rng = Rng::new(0);
    let (l1, l2) = (GcnConv::new(8, 8, &rng), GcnConv::new(8, 2, &rng));
    let mut opt = Adam::new([l1.parameters(), l2.parameters()].concat(), 0.05);
    let mut losses = Vec::new();
    for _ in 0..60 {
        opt.zero_grad();
        let logits = l2.forward(&l1.forward(&x, &e).unwrap().relu(), &e).unwrap();
        let loss = cross_entropy_indices(&logits, &labels).unwrap();
        losses.push(loss.item());
        loss.backward();
        opt.step();
    }
    assert!(losses[59] < 0.1 * losses[0], "loss {} -> {}", losses[0], losses[59]);
    let logits = l2.forward(&l1.forward(&x, &e).unwrap().relu(), &e).unwrap().to_vec();
    for n in 0..8 { assert_eq!((logits[n * 2 + 1] > logits[n * 2]) as usize, n / 4, "node {n}"); }
}
