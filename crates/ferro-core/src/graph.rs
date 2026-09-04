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

use crate::dispatch::{backend_for, ChainStepRef, OpTag};
use crate::error::{Error, Result};
use crate::shape::numel;
use crate::tensor::{device_leaf, raw_binary_k, raw_unary_k, Tensor};

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
    /// Kernel tag recorded on this node's op (kind-routed ops only).
    pub tag: Option<OpTag>,
    /// Tensor ids of this node's recorded op inputs, in op order.
    pub inputs: Vec<usize>,
}

pub struct Graph {
    pub nodes: HashMap<usize, GraphNode>,
    /// All reachable node ids, roots first (reverse topological order, the
    /// same direction `backward_with` walks).
    pub order: Vec<usize>,
    /// The captured tensors keyed by node id, so chain resolution can hand
    /// real operands to an executor without re-running anything. Holds one
    /// Arc clone per recorded node for as long as the Graph lives.
    pub tensors: HashMap<usize, Tensor>,
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
        let mut tensors: HashMap<usize, Tensor> = HashMap::new();
        let mut order = Vec::with_capacity(postorder.len());
        for t in postorder.into_iter().rev() {
            let owned: Vec<Tensor> =
                t.0.op
                    .as_ref()
                    .map(|op| op.inputs().to_vec())
                    .unwrap_or_default();
            let refs: Vec<&Tensor> = owned.iter().collect();
            let mut kind = classify(&t, &refs);
            let tag = t.0.op.as_ref().and_then(|op| op.tag);
            // `classify` guesses kind from shapes alone, so a same-shape
            // elementwise binary (e.g. 512x512 * 512x512) can satisfy the
            // matmul shape contract and be mislabelled MatMul -- which is not
            // fusible, silently breaking a pointwise chain. The op TAG is
            // ground truth: matmul records untagged (`record_fn`), every
            // pointwise op records `record_fn_tagged`. Reconcile: a Binary tag
            // forces Binary, a Unary tag forces Unary.
            match tag {
                Some(OpTag::Binary(_)) => kind = NodeKind::Binary,
                Some(OpTag::Unary(_)) => kind = NodeKind::Unary,
                None => {}
            }
            tensors.insert(t.id(), t.clone());
            let node = GraphNode {
                id: t.id(),
                shape: t.shape().to_vec(),
                kind,
                tag,
                inputs: owned.iter().map(|i| i.id()).collect(),
            };
            order.push(node.id);
            nodes.insert(node.id, node);
        }
        Graph {
            nodes,
            order,
            tensors,
        }
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

    /// Evaluate this captured graph through the fusion plan, sourcing every
    /// leaf value from the tensors captured at `from_root` time (no re-run of
    /// the original tape). Pointwise chains the planner found execute as ONE
    /// fused `chain_dev` launch each; every other node runs its single raw
    /// kernel. Returns a detached tensor equal to the eager root but produced
    /// with `launches_after` launches instead of one-per-op.
    ///
    /// This is the eager-fusion seam: a caller that just built `y = relu(x)*a+b`
    /// can hand the root's graph here and get the fused result, without the
    /// `Replay::capture(build)` closure dance (whose leaves must be re-supplied).
    pub fn eval_fused(&self) -> Result<Tensor> {
        let mut values: HashMap<usize, Tensor> = HashMap::new();
        // Seed leaves from the captured tensors: a leaf is any node the walk
        // recorded with no op (its value is an input, already realised).
        for (&id, node) in &self.nodes {
            if node.kind == NodeKind::Leaf {
                let t = self.tensors.get(&id).ok_or_else(|| Error::Unsupported {
                    op: "eval_fused",
                    msg: format!("leaf {id} has no captured tensor"),
                })?;
                values.insert(id, t.clone());
            }
        }
        let plan = self.plan_fusion();
        let mut chain_of: HashMap<usize, usize> = HashMap::new();
        for (ci, c) in plan.chains.iter().enumerate() {
            for &n in &c.nodes {
                chain_of.insert(n, ci);
            }
        }
        let mut done_chains = vec![false; plan.chains.len()];
        // `order` is roots-first; evaluate producers before consumers.
        for &id in self.order.iter().rev() {
            if values.contains_key(&id) {
                continue;
            }
            match self.nodes[&id].kind {
                NodeKind::Leaf => {
                    return Err(Error::Unsupported {
                        op: "eval_fused",
                        msg: format!("leaf {id} was not seeded"),
                    })
                }
                NodeKind::Reduce | NodeKind::MatMul | NodeKind::Other => {
                    let out = self.eval_raw_node(id, &values)?;
                    values.insert(id, out);
                }
                NodeKind::Unary | NodeKind::Binary => match chain_of.get(&id) {
                    Some(&ci) if !done_chains[ci] => {
                        done_chains[ci] = true;
                        let chain = &plan.chains[ci];
                        let exec = chain.resolve(self)?;
                        let out = chain.run(&exec)?;
                        values.insert(*chain.nodes.last().expect("non-empty chain"), out);
                    }
                    Some(_) => {}
                    None => {
                        let out = self.eval_raw_node(id, &values)?;
                        values.insert(id, out);
                    }
                },
            }
        }
        let root = *self.order.first().ok_or_else(|| Error::Unsupported {
            op: "eval_fused",
            msg: "empty graph".into(),
        })?;
        values.get(&root).cloned().ok_or_else(|| Error::Unsupported {
            op: "eval_fused",
            msg: "root not evaluated".into(),
        })
    }

    /// Evaluate one tagged Unary/Binary node from already-computed inputs.
    fn eval_raw_node(&self, id: usize, values: &HashMap<usize, Tensor>) -> Result<Tensor> {
        let node = &self.nodes[&id];
        let tag = node.tag.ok_or_else(|| Error::Unsupported {
            op: "eval_fused",
            msg: format!("node {id} has no kernel tag and cannot be replayed"),
        })?;
        let ins: Vec<Tensor> = node
            .inputs
            .iter()
            .map(|i| {
                values.get(i).cloned().ok_or_else(|| Error::Unsupported {
                    op: "eval_fused",
                    msg: format!("input {i} of node {id} not yet evaluated"),
                })
            })
            .collect::<Result<_>>()?;
        match tag {
            OpTag::Unary(kind) => raw_unary_k(&ins[0], kind),
            OpTag::Binary(kind) => raw_binary_k("eval_fused", &ins[0], &ins[1], kind),
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

/// A resolved chain ready to execute in one backend call.
pub struct ExecutableChain {
    pub steps: Vec<ChainStepRef>,
    /// Operand tensors: index 0 is the seed; later entries are the buffers
    /// referenced by Binary/BinaryBc `other` indices.
    pub operands: Vec<Tensor>,
    pub out_shape: Vec<usize>,
}

fn padded_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let pad = out_shape.len() - shape.len();
    let mut strides = vec![0usize; out_shape.len()];
    let mut acc = 1usize;
    for d in (0..shape.len()).rev() {
        strides[d + pad] = if shape[d] == 1 { 0 } else { acc };
        if shape[d] != 1 {
            acc *= shape[d];
        }
    }
    strides
}

impl FusedChain {
    /// Resolve this chain's node ids into executable steps over the graph's
    /// captured tensors. Every node must carry an op tag (the planner only
    /// admits tagged Unary/Binary nodes, but the Graph can be built from any
    /// tape, so this stays fallible). The seed is the chain's first node;
    /// each subsequent step reads its predecessor plus any second operand
    /// captured here by index. Broadcast shapes are decomposed against the
    /// SEED's shape (all chain intermediates share it pointwise).
    pub fn resolve(&self, g: &Graph) -> Result<ExecutableChain> {
        let mut steps: Vec<ChainStepRef> = Vec::with_capacity(self.nodes.len());
        // operand index -> tensor id; slot 0 is filled with the seed below.
        let mut slots: Vec<usize> = Vec::new();
        for (k, &id) in self.nodes.iter().enumerate() {
            let node = &g.nodes[&id];
            let tag = node.tag.ok_or_else(|| Error::Unsupported {
                op: "chain_resolve",
                msg: format!("node {id} has no kernel tag and cannot be fused"),
            })?;
            match (tag, k == 0) {
                (OpTag::Unary(kind), true) => {}
                (OpTag::Unary(kind), false) => {
                    steps.push(ChainStepRef::Unary(kind));
                }
                (OpTag::Binary(kind), _) => {
                    let pred = self.nodes[k - 1];
                    let other_id = *node.inputs.iter().find(|&&i| i != pred).ok_or_else(|| {
                        Error::Unsupported {
                            op: "chain_resolve",
                            msg: format!("binary node {id} has no second input"),
                        }
                    })?;
                    let slot = match slots.iter().position(|&s| s == other_id) {
                        Some(s) => s,
                        None => {
                            slots.push(other_id);
                            slots.len() - 1
                        }
                    };
                    // +1: slot 0 of the kernel signature is the seed itself.
                    let other = slot + 1;
                    let same = g.tensors[&other_id].shape() == g.tensors[&self.nodes[0]].shape();
                    if same {
                        steps.push(ChainStepRef::Binary { kind, other });
                    } else {
                        let out_shape = g.tensors[&id].shape().to_vec();
                        let in_shape = g.tensors[&other_id].shape().to_vec();
                        steps.push(ChainStepRef::BinaryBc {
                            kind,
                            dims: out_shape.iter().map(|&d| d as u32).collect(),
                            strides: padded_strides(&in_shape, &out_shape)
                                .iter()
                                .map(|&d| d as u32)
                                .collect(),
                            other,
                        });
                    }
                }
                _ => {
                    return Err(Error::Unsupported {
                        op: "chain_resolve",
                        msg: "a chain cannot start with a binary step".into(),
                    })
                }
            }
        }
        let mut operands = vec![g.tensors[&self.nodes[0]].clone()];
        for id in &slots {
            operands.push(g.tensors[id].clone());
        }
        Ok(ExecutableChain {
            steps,
            operands,
            out_shape: g.nodes[&self.nodes[0]].shape.clone(),
        })
    }

    /// Evaluate the chain on its seed's device: one fused backend launch when
    /// the device is resident and the backend implements `chain_dev`, else a
    /// sequential per-op fallback that computes exactly the same math through
    /// the ordinary raw kernels. Returns a detached tensor.
    pub fn run(&self, chain: &ExecutableChain) -> Result<Tensor> {
        run_chain(chain)
    }
}

/// Execute a resolved chain on its seed's device: one fused backend launch
/// when every operand is device-resident and the backend implements
/// `chain_dev`, else the sequential host fallback. Free-standing so a
/// precompiled [`CompiledChain`] handle can replay without re-walking a tape.
pub fn run_chain(chain: &ExecutableChain) -> Result<Tensor> {
        let seed = &chain.operands[0];
        let all_resident = chain.operands.iter().all(|t| t.device_resident_whole());
        if all_resident {
            if let Ok(out) = (|| {
                let backend = backend_for(seed.device())?;
                // One read guard per distinct StorageCell, ACQUIRED in
                // global address order (not operand order): operands can
                // repeat a tensor (a same-thread double read of one lock
                // can itself deadlock behind a queued writer, so dedup
                // first), and two chains sharing operands in reversed order
                // would otherwise lock them in opposite orders - the same
                // AB-BA hazard fixed in `tensor::PairGuard`, once any writer
                // (an in-place op) can queue on either lock. Two passes:
                // collect the distinct (tensor, pointer) pairs, sort by
                // pointer, THEN lock in that order.
                let mut by_ptr: Vec<(*const crate::tensor::StorageCell, &Tensor)> = Vec::new();
                for t in &chain.operands {
                    let p = std::sync::Arc::as_ptr(&t.0.storage);
                    if !by_ptr.iter().any(|&(q, _)| q == p) {
                        by_ptr.push((p, t));
                    }
                }
                by_ptr.sort_unstable_by_key(|&(p, _)| p);
                let mut guards: Vec<(
                    *const crate::tensor::StorageCell,
                    std::sync::RwLockReadGuard<crate::tensor::Storage>,
                )> = Vec::with_capacity(by_ptr.len());
                for (p, t) in by_ptr {
                    guards.push((p, t.0.storage.read()));
                }
                let bufs: Vec<&dyn crate::dispatch::DeviceBuffer> = chain
                    .operands
                    .iter()
                    .map(|t| -> &dyn crate::dispatch::DeviceBuffer {
                        let p = std::sync::Arc::as_ptr(&t.0.storage);
                        let (_, g) = guards.iter().find(|(q, _)| *q == p).unwrap();
                        match &**g {
                            crate::tensor::Storage::Device(b) => b.as_ref(),
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                backend.chain_dev(&chain.steps, &bufs)
            })() {
                return Ok(device_leaf(out, &chain.out_shape, seed.device()));
            }
        }
        run_chain_host(chain)
    }

/// Sequential per-op fallback for a resolved chain: same math as the fused
/// kernel through ordinary raw kernels. Used when operands are not all
/// device-resident or the backend lacks `chain_dev`.
pub fn run_chain_host(chain: &ExecutableChain) -> Result<Tensor> {
        let mut cur = chain.operands[0].clone();
        let mut slot_values: HashMap<usize, Tensor> = HashMap::new();
        // `resolve` sets each binary step's `other` to the operand's index in
        // `chain.operands` (operands[0] is the seed; operands[k] is referenced
        // as other==k). So map operand index -> tensor directly; the earlier
        // `slot+1` shifted every key by one and missed on lookup.
        for (idx, t) in chain.operands.iter().enumerate().skip(1) {
            slot_values.insert(idx, t.clone());
        }
        for step in &chain.steps {
            cur = match step {
                ChainStepRef::Unary(kind) => raw_unary_k(&cur, *kind)?,
                ChainStepRef::Binary { kind, other } => {
                    raw_binary_k("chain", &cur, &slot_values[other], *kind)?
                }
                ChainStepRef::BinaryBc { kind, other, .. } => {
                    raw_binary_k("chain_bc", &cur, &slot_values[other], *kind)?
                }
            };
        }
        Ok(cur)
}

/// A precompiled fused chain: resolve the fusion plan ONCE from a tape, then
/// replay the single fused kernel many times without re-walking the tape or
/// re-planning. This is what turns the fused-kernel throughput win into a
/// Python-visible speedup - the eager `.fuse()` path re-plans every call, and
/// that host cost swamps the DRAM saving on a memory-bound chain.
///
/// The captured operands are immutable (ferro tensors never mutate storage
/// identity), so replay does identical device traffic each call; it is a
/// faithful timing of one fused launch vs the equivalent eager launches.
pub struct CompiledChain {
    exec: ExecutableChain,
}

impl CompiledChain {
    /// Build a handle from a graph root: walk the tape once, plan fusion, and
    /// resolve the single chain that produces the root. Errors if the root's
    /// producing region is not a single fusible pointwise chain (e.g. it
    /// contains a matmul/reduce, or nothing fusible).
    pub fn compile(root: &Tensor) -> Result<CompiledChain> {
        let g = Graph::from_root(root);
        let plan = g.plan_fusion();
        let root_id = *g.order.first().ok_or_else(|| Error::Unsupported {
            op: "compile_chain",
            msg: "empty graph".into(),
        })?;
        let chain = plan
            .chains
            .iter()
            .find(|c| c.nodes.last() == Some(&root_id))
            .ok_or_else(|| Error::Unsupported {
                op: "compile_chain",
                msg: "root is not produced by a single fusible pointwise chain".into(),
            })?;
        let exec = chain.resolve(&g)?;
        Ok(CompiledChain { exec })
    }

    /// Replay the fused chain: one backend `chain_dev` launch when resident.
    pub fn replay(&self) -> Result<Tensor> {
        run_chain(&self.exec)
    }

    /// Number of operands (seed + distinct second operands) captured.
    pub fn num_operands(&self) -> usize {
        self.exec.operands.len()
    }

    /// Number of fused steps (chain length after the seed).
    pub fn num_steps(&self) -> usize {
        self.exec.steps.len()
    }
}

#[doc(hidden)]
pub fn _dbg_op_present(t: &Tensor) -> bool {
    t.0.op.is_some()
}

#[doc(hidden)]
pub fn _dbg_requires_grad(t: &Tensor) -> bool {
    t.0.requires_grad
}

#[doc(hidden)]
pub fn _dbg_graph_shape(t: &Tensor) -> (bool, bool, usize) {
    // (root has op, root requires_grad, inputs of root's op)
    (
        t.0.op.is_some(),
        t.0.requires_grad,
        t.0.op.as_ref().map(|o| o.inputs().len()).unwrap_or(0),
    )
}

#[doc(hidden)]
pub fn _dbg_uses_grad(t: &Tensor) -> usize {
    // count how many tensors in the local chain have requires_grad
    if let Some(op) = &t.0.op {
        op.inputs().iter().filter(|i| i.0.requires_grad).count() * 100 + op.inputs().len()
    } else { 0 }
}
