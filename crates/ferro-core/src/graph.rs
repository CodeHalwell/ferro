//! Graph compiler v0: DAG analysis over the autograd tape and a pointwise
//! fusion planner. Read-only: this module never touches `record_fn` or any op
//! path. It walks the op graph that ordinary forward execution already
//! recorded (the same structure `Tensor::backward` walks) and classifies each
//! recorded node structurally, since `Op` carries no op-name tag.
//!
//! Classification is heuristic by design (v0):
//!   - 0 recorded inputs            -> Leaf
//!   - 1 input, fewer output elems  -> Reduce (fusion barrier)
//!   - 1 input, same element count  -> Unary
//!   - 2 inputs contracting on the inner dim -> MatMul (fusion barrier)
//!   - other                        -> Other / Binary
//!
//! The planner finds maximal linear runs of Unary/Binary nodes whose
//! intermediates have exactly one consumer; each such run of length n could be
//! compiled to a single kernel, saving n-1 launches per pass (a backward pass
//! roughly doubles that, since each fused forward link also fuses its VJP
//! links). Wave 4 will consume the plan; v0 only reports it.
//!
//! # How nvrtc generation would consume a FusionPlan (wave 4 sketch)
//!
//! `ferro-cuda/src/kernels.rs` already generates CUDA C source as pure string
//! work: `unary_expr`/`binary_expr` produce per-op expressions,
//! `unary_source`/`binary_source`/`binary_bc_source` wrap them into one
//! `extern "C" __global__ void ferro_kernel(...)`, compiled by nvrtc through
//! `CudaBackend::get_kernel`, whose cache keys on source text. A fused chain
//! compiles the same way: emit one kernel whose body threads the existing
//! expression builders together -
//!
//! ```text
//! float v0 = a[i];
//! float v1 = <unary_expr(gelu)>(v0);        // intermediate stays in a register
//! out[i]   = <binary_expr(add)>(v1, b[i]); // broadcast offsets from broadcast_strides
//! ```
//!
//! i.e. concatenate expressions instead of launching one kernel per op,
//! eliminating the global-memory round trip per intermediate. The generated
//! source (hence the `get_kernel` cache key) is derived deterministically from
//! the chain's node kinds and ranks, exactly as `binary_bc_source(kind, rank)`
//! keys today. Barrier nodes (MatMul, Reduce, Other) keep their hand-written
//! kernels: the planner's chain boundaries are precisely the launch boundaries
//! of the compiled schedule.

use std::collections::{HashMap, HashSet};

use crate::shape::numel;
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Leaf,
    Unary,
    Binary,
    Reduce,
    MatMul,
    Other,
}

pub struct GraphNode {
    pub id: usize,
    pub shape: Vec<usize>,
    pub kind: NodeKind,
    /// Tensor ids of this node's recorded op inputs, in op order.
    pub inputs: Vec<usize>,
}

pub struct Graph {
    pub nodes: HashMap<usize, GraphNode>,
    /// All reachable node ids, roots first (reverse topological order, the
    /// same direction `backward_with` walks).
    pub order: Vec<usize>,
}

/// One fusible pointwise run: node ids in execution order, every interior
/// link Unary/Binary with the previous node as sole producer and itself as
/// its only consumer.
#[derive(Debug)]
pub struct FusedChain {
    pub nodes: Vec<usize>,
}

/// v0 output: the fusion opportunities found in a captured graph. Applying it
/// is wave 4; today this only estimates launch savings.
#[derive(Debug)]
pub struct FusionPlan {
    pub chains: Vec<FusedChain>,
    /// Kernel launches the current schedule issues for all recorded (non-leaf)
    /// nodes.
    pub launches_before: usize,
    /// Launches after applying every chain in `chains`.
    pub launches_after: usize,
}

impl FusionPlan {
    pub fn launches_saved(&self) -> usize {
        self.launches_before - self.launches_after
    }
}

fn classify(out: &Tensor, inputs: &[&Tensor]) -> NodeKind {
    match inputs.len() {
        0 => NodeKind::Leaf,
        1 => {
            if numel(out.shape()) < numel(inputs[0].shape()) {
                NodeKind::Reduce
            } else {
                NodeKind::Unary
            }
        }
        2 => {
            let (a, b) = (inputs[0], inputs[1]);
            let (ar, br) = (a.shape().len(), b.shape().len());
            let contracts = br >= 2 && ar >= 1 && a.shape()[ar - 1] == b.shape()[br - 2];
            let head = br.saturating_sub(2);
            let mut expected: Vec<usize> = a.shape()[..ar.saturating_sub(1)].to_vec();
            expected.extend_from_slice(&b.shape()[..head]);
            expected.extend_from_slice(&b.shape()[br - 1..]);
            if contracts && out.shape() == expected.as_slice() {
                NodeKind::MatMul
            } else {
                NodeKind::Binary
            }
        }
        _ => NodeKind::Other,
    }
}

impl Graph {
    /// Walk the tape rooted at `build()`'s return value. Typically called
    /// after `.backward()` ("post-backward"), but walking only reads recorded
    /// op links, which exist as soon as forward execution does.
    pub fn capture<F: FnOnce() -> Tensor>(build: F) -> Graph {
        Graph::from_root(&build())
    }

    pub fn from_root(root: &Tensor) -> Graph {
        // Post-order DFS with an explicit stack (same shape as autograd's
        // build_topo), then reversed into forward order.
        let mut seen: HashSet<usize> = HashSet::new();
        let mut postorder: Vec<Tensor> = Vec::new();
        seen.insert(root.id());
        stack_walk(root, &mut seen, &mut postorder);

        let mut nodes = HashMap::new();
        let mut order = Vec::with_capacity(postorder.len());
        for t in postorder.into_iter().rev() {
            let owned: Vec<Tensor> =
                t.0.op
                    .as_ref()
                    .map(|op| op.inputs().to_vec())
                    .unwrap_or_default();
            let refs: Vec<&Tensor> = owned.iter().collect();
            let kind = classify(&t, &refs);
            let node = GraphNode {
                id: t.id(),
                shape: t.shape().to_vec(),
                kind,
                inputs: owned.iter().map(|i| i.id()).collect(),
            };
            order.push(node.id);
            nodes.insert(node.id, node);
        }
        Graph { nodes, order }
    }

    fn consumer_counts(&self) -> HashMap<usize, usize> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for id in &self.order {
            for inp in &self.nodes[id].inputs {
                *counts.entry(*inp).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Report the pointwise-chain fusion plan. A chain starts right after a
    /// leaf or barrier node and extends through Unary/Binary nodes while each
    /// intermediate feeds exactly one consumer whose first recorded input is
    /// that intermediate (linear chain: no fan-out, no barrier between).
    pub fn plan_fusion(&self) -> FusionPlan {
        let consumers = self.consumer_counts();
        let by_id = |id: usize| &self.nodes[&id];
        let fusible = |n: &GraphNode| matches!(n.kind, NodeKind::Unary | NodeKind::Binary);
        let mut chains: Vec<FusedChain> = Vec::new();
        let mut used: HashSet<usize> = HashSet::new();

        for &start in self.order.iter().rev() {
            if !fusible(by_id(start)) || used.contains(&start) {
                continue;
            }
            // A chain STARTS here only when its producer cannot continue a
            // chain into it: leaf/barrier producer, or an intermediate with
            // more than one consumer.
            let node = by_id(start);
            let starts = node.inputs.first().map_or(true, |&p| {
                !fusible(by_id(p)) || consumers.get(&p) != Some(&1)
            });
            if !starts {
                continue;
            }
            let mut run = vec![start];
            loop {
                let tail = run[run.len() - 1];
                let single_consumer = consumers.get(&tail) == Some(&1);
                let next = single_consumer.then(|| {
                    self.order
                        .iter()
                        .filter(|&&id| !used.contains(&id))
                        .filter_map(|&id| {
                            let n = by_id(id);
                            (fusible(n) && n.inputs.first() == Some(&tail)).then_some(id)
                        })
                        .next()
                });
                match next.flatten() {
                    Some(id) => run.push(id),
                    None => break,
                }
            }
            if run.len() > 1 {
                for &id in &run {
                    used.insert(id);
                }
                chains.push(FusedChain { nodes: run });
            } else {
                used.insert(start);
            }
        }

        let non_leaf = self.order.len()
            - self
                .nodes
                .values()
                .filter(|n| n.kind == NodeKind::Leaf)
                .count();
        let fused_away: usize = chains.iter().map(|c| c.nodes.len() - 1).sum();
        FusionPlan {
            chains,
            launches_before: non_leaf,
            launches_after: non_leaf - fused_away,
        }
    }
}

// Separated so the DFS stack lives even when the compiler cannot prove the
// recursion depth; iterative by construction like autograd's build_topo.
fn stack_walk(root: &Tensor, seen: &mut HashSet<usize>, postorder: &mut Vec<Tensor>) {
    let mut stack: Vec<(Tensor, usize)> = vec![(root.clone(), 0)];
    while let Some((t, i)) = stack.pop() {
        if let Some(op) = &t.0.op {
            let inputs = op.inputs();
            if i < inputs.len() {
                let child = inputs[i].clone();
                stack.push((t.clone(), i + 1));
                if seen.insert(child.id()) {
                    stack.push((child, 0));
                }
                continue;
            }
        }
        postorder.push(t);
    }
}
