//! Replay executor: run a recorded forward through the fusion plan instead of
//! op-by-op. `Replay::capture` records the tape once (the ordinary eager ops
//! still execute, producing both values and the graph); `replay` re-derives
//! every node from the leaves by executing fused chains where the planner
//! found them and single raw kernels everywhere else, so a replayed forward
//! issues one launch per chain rather than one per op.
//!
//! Opt-in and detached: replay produces plain values (no autograd graph).
//! It is the seam CUDA-graph capture will wrap in wave 5 - capture needs an
//! execution schedule that is already chain-shaped.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::graph::{Graph, NodeKind};
use crate::tensor::{raw_binary_k, raw_unary_k, Tensor};

pub struct Replay {
    graph: Graph,
    /// Leaf ids in walk order: the tensors a caller must supply.
    pub leaves: Vec<usize>,
}

impl Replay {
    /// Capture a tape by running `build` once eagerly (values are computed
    /// exactly as today) and keeping its graph.
    pub fn capture<F: FnOnce() -> Tensor>(build: F) -> Replay {
        let root = build();
        let mut graph = Graph::from_root(&root);
        // from_root's order is roots-first; replay walks leaves-first, so
        // reverse it. The root is then the LAST id.
        graph.order.reverse();
        let leaves = graph
            .order
            .iter()
            .copied()
            .filter(|&id| graph.nodes[&id].kind == NodeKind::Leaf)
            .collect();
        Replay { graph, leaves }
    }

    pub fn plan_launches(&self) -> (usize, usize) {
        let p = self.graph.plan_fusion();
        (p.launches_before, p.launches_after)
    }

    /// Re-execute the tape from supplied leaf values (order matches
    /// `self.leaves`). Chains run as one fused backend call when every
    /// operand is resident and the backend implements chain_dev; everything
    /// else runs raw per-op kernels with identical math.
    pub fn replay(&self, leaves: &[Tensor]) -> Result<Tensor> {
        if leaves.len() != self.leaves.len() {
            return Err(Error::InvalidShape {
                op: "replay",
                msg: format!(
                    "expected {} leaf tensors, got {}",
                    self.leaves.len(),
                    leaves.len()
                ),
            });
        }
        let mut values: HashMap<usize, Tensor> = HashMap::new();
        for (&id, t) in self.leaves.iter().zip(leaves) {
            values.insert(id, t.clone());
        }
        let plan = self.graph.plan_fusion();
        let mut chain_of: HashMap<usize, usize> = HashMap::new();
        for (ci, c) in plan.chains.iter().enumerate() {
            for &n in &c.nodes {
                chain_of.insert(n, ci);
            }
        }
        let mut done_chains = vec![false; plan.chains.len()];
        for &id in &self.graph.order.clone() {
            if values.contains_key(&id) {
                continue;
            }
            let kind = self.graph.nodes[&id].kind;
            match kind {
                NodeKind::Leaf => unreachable!("leaf without a supplied value"),
                NodeKind::Reduce | NodeKind::MatMul | NodeKind::Other => {
                    let out = eval_node(&self.graph, id, &values)?;
                    values.insert(id, out);
                }
                NodeKind::Unary | NodeKind::Binary => match chain_of.get(&id) {
                    Some(&ci) if !done_chains[ci] => {
                        done_chains[ci] = true;
                        let chain = &plan.chains[ci];
                        let exec = chain.resolve(&self.graph)?;
                        let out = chain.run(&exec)?;
                        values.insert(*chain.nodes.last().expect("non-empty chain"), out);
                    }
                    Some(_) => {}
                    None => {
                        let out = eval_node(&self.graph, id, &values)?;
                        values.insert(id, out);
                    }
                },
            }
        }
        let root = *self.graph.order.last().ok_or_else(|| Error::Unsupported {
            op: "replay",
            msg: "empty graph".into(),
        })?;
        Ok(values[&root].clone())
    }
}

fn eval_node(g: &Graph, id: usize, values: &HashMap<usize, Tensor>) -> Result<Tensor> {
    let node = &g.nodes[&id];
    let tag = node.tag.ok_or_else(|| Error::Unsupported {
        op: "replay",
        msg: format!("node {id} has no kernel tag"),
    })?;
    let ins: Vec<Tensor> = node.inputs.iter().map(|&i| values[&i].clone()).collect();
    match tag {
        crate::dispatch::OpTag::Unary(kind) => raw_unary_k(&ins[0], kind),
        crate::dispatch::OpTag::Binary(kind) => raw_binary_k("replay", &ins[0], &ins[1], kind),
    }
}
