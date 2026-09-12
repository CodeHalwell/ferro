//! Graph analysis and compiled replay over recorded operation links.
//!
//! `Graph::plan_fusion` is a structural, heuristic analysis tool. Execution
//! uses `CompiledChain`: tagged pointwise nodes and explicit forward metadata replay;
//! unsupported operations are rejected before any replay work is performed.
//! Compilation captures current graph leaves and schedules single-consumer,
//! left-input chains in dependency order. Shared intermediates are evaluated
//! once. Shape expansion is a chain boundary and uses ordinary broadcast
//! dispatch, including resident CUDA kernels when available.

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
            expected.extend_from_slice(&b.shape()[br.saturating_sub(1)..]);
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

    /// Replay tagged operations from current leaves using the compiled schedule.
    /// Operations without forward metadata are rejected rather than frozen.
    pub fn eval_fused(&self) -> Result<Tensor> {
        let id = self.order.first().ok_or_else(|| Error::Unsupported {
            op: "eval_fused", msg: "empty graph".into(),
        })?;
        let root = &self.tensors[id];
        if self.nodes[id].inputs.is_empty() { return Ok(root.detach_copy()); }
        CompiledChain::compile(root)?.replay()
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
        let seed_id = *self.nodes.first().ok_or_else(|| Error::Unsupported {
            op: "chain_resolve", msg: "empty chain".into(),
        })?;
        if self.nodes.iter().any(|id| g.nodes[id].shape != g.nodes[&seed_id].shape) {
            return Err(Error::Unsupported {
                op: "chain_resolve", msg: "seed-expanding broadcast is not supported".into(),
            });
        }
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
                (_, true) => {}
                (OpTag::Unary(kind), false) => {
                    steps.push(ChainStepRef::Unary(kind));
                }
                (OpTag::Binary(kind), false) => {
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
/// `chain_dev`, else sequential raw dispatch. Free-standing so a
/// precompiled [`CompiledChain`] handle can replay without re-walking a tape.
pub fn run_chain(chain: &ExecutableChain) -> Result<Tensor> {
        let seed = &chain.operands[0];
        // The chain ABI indexes its seed linearly and sizes output from it.
        // Expanding seeds must use the ordinary broadcast device kernel.
        let all_resident = seed.shape() == chain.out_shape
            && chain.operands.iter().all(|t| t.device() == seed.device() && t.device_resident_whole());
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
/// kernel through ordinary raw kernels (which can remain device-resident).
/// Used when operands are not all
/// device-resident or the backend lacks `chain_dev`.
pub fn run_chain_host(chain: &ExecutableChain) -> Result<Tensor> {
        let mut cur = chain.operands[0].clone();
        let mut slot_values: HashMap<usize, Tensor> = HashMap::new();
        // `resolve` sets each binary step's `other` to the operand's index in
        // `chain.operands` (operands[0] is the seed; operands[k] is referenced
        // as other==k). So map operand index -> tensor directly; the earlier
        // `slot+1` shifted every key by one and missed on lookup. Slot zero
        // is valid too: a compiled chain can reuse its original seed leaf.
        for (idx, t) in chain.operands.iter().enumerate() {
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

/// A precompiled inference DAG: resolve once from a tape, then replay without
/// re-walking or re-planning. Compilation is separate from inference timing.
///
/// Captures graph leaves, not computed intermediates, and reads their current
/// storage on every replay. Storage identity is stable; values may change.
/// Build with `capture(|| expression)` to record inference without gradients.
/// Outside capture, detached/no-grad computations are opaque graph input values.
pub struct CompiledChain {
    leaves: HashMap<usize, Tensor>,
    schedule: Vec<CompiledRun>,
    root: usize,
    steps: usize,
}

struct CompiledRun {
    forward: Option<crate::autograd::ForwardOp>,
    output: usize,
    inputs: Vec<usize>,
    steps: Vec<ChainStepRef>,
    shape: Vec<usize>,
}

/// Static replay is exclusive; only explicitly copied snapshots escape.
pub struct PreparedChain {
    execution: Box<dyn crate::dispatch::StaticExecution>,
    leaves: Vec<Tensor>,
    shape: Vec<usize>,
    device: crate::Device,
}

impl PreparedChain {
    pub fn replay(&mut self) -> Result<()> {
        // Device mutation uses shared storage guards. Exclusive guards here
        // prevent updates interleaving individual boundary enqueues.
        let guards: Vec<_> = self.leaves.iter().map(|t| t.0.storage.write()).collect();
        let result = self.execution.replay();
        drop(guards);
        result
    }
    pub fn snapshot(&self) -> Result<Tensor> {
        Ok(device_leaf(self.execution.snapshot()?, &self.shape, self.device))
    }
    pub fn replay_count(&self) -> usize { self.execution.replay_count() }
}

impl CompiledChain {
    pub fn prepare_static(&self) -> Result<PreparedChain> {
        use crate::dispatch::StaticRun;
        use crate::tensor::Storage;
        use std::sync::Arc;
        let unsupported = |msg: &str| Error::Unsupported { op: "prepare_static", msg: msg.into() };
        let effective_inputs = |run: &CompiledRun| {
            let mut inputs = run.inputs.clone();
            if let Some(crate::autograd::ForwardOp::LayerNorm {weight,bias,..}) = &run.forward {
                if !weight { inputs[1] = inputs[0]; }
                if !bias { inputs[2] = inputs[0]; }
            }
            inputs
        };
        let used: HashSet<_> = self.schedule.iter().flat_map(effective_inputs).collect();
        let mut ids: Vec<_> = self.leaves.keys().filter(|id| used.contains(id)).copied().collect();
        ids.sort_unstable();
        let leaves: Vec<_> = ids.iter().map(|id| self.leaves[id].clone()).collect();
        let device = leaves[0].device();
        if leaves.iter().any(|t| t.device() != device || !t.device_resident_whole() || t.requires_grad()) {
            return Err(unsupported("requires whole resident inference leaves on one device"));
        }
        let mut slots: HashMap<_, _> = ids.into_iter().enumerate().map(|(i,id)| (id,i)).collect();
        let mut shapes: Vec<_> = leaves.iter().map(|t| t.shape().to_vec()).collect();
        let mut runs = Vec::new();
        for run in &self.schedule {
            let mut inputs: Vec<_> = effective_inputs(run).iter().map(|id| slots[id]).collect();
            let mut static_shape = run.shape.clone();
            let op = match &run.forward {
                Some(crate::autograd::ForwardOp::MatMul) => {
                    let a = &shapes[inputs[0]];
                    let b = &shapes[inputs[1]];
                    if a.len() != 2 || b.len() != 2 || a[1] != b[0] { return Err(unsupported("invalid static matmul shape")); }
                    crate::dispatch::StaticOp::MatMul { m: a[0], k: a[1], n: b[1] }
                },
                Some(crate::autograd::ForwardOp::Bmm) => {
                    let a = &shapes[inputs[0]]; let b = &shapes[inputs[1]];
                    if a.len()!=3 || b.len()!=3 || a[0]!=b[0] || a[2]!=b[1] { return Err(unsupported("invalid static bmm shape")); }
                    crate::dispatch::StaticOp::Bmm { batch:a[0],m:a[1],k:a[2],n:b[2] }
                },
                Some(crate::autograd::ForwardOp::Reshape(_)) => crate::dispatch::StaticOp::Layout { strides: crate::shape::default_strides(&run.shape) },
                Some(crate::autograd::ForwardOp::Transpose(a,b)) => {
                    let mut strides = crate::shape::default_strides(&shapes[inputs[0]]);
                    strides.swap(*a,*b);
                    crate::dispatch::StaticOp::Layout { strides }
                },
                Some(crate::autograd::ForwardOp::Softmax(dim)) => {
                    let shape = shapes[inputs[0]].clone();
                    let product = |s: &[usize]| s.iter().try_fold(1usize, |a,&b|a.checked_mul(b)).ok_or_else(||unsupported("static softmax shape overflow"));
                    let (outer,cols,inner) = (product(&shape[..*dim])?,shape[*dim],product(&shape[*dim+1..])?);
                    if *dim+1 == shape.len() {
                        crate::dispatch::StaticOp::Softmax { rows:outer,cols }
                    } else {
                        let rows = outer.checked_mul(inner).ok_or_else(||unsupported("static softmax shape overflow"))?;
                        let width = cols.checked_mul(inner).ok_or_else(||unsupported("static softmax shape overflow"))?;
                        let permuted = vec![outer,inner,cols];
                        runs.push(StaticRun { inputs:inputs.clone(),op:crate::dispatch::StaticOp::Layout { strides:vec![width,1,inner] },shape:permuted.clone() });
                        inputs[0] = shapes.len();
                        shapes.push(permuted.clone());
                        runs.push(StaticRun { inputs:inputs.clone(),op:crate::dispatch::StaticOp::Softmax { rows,cols },shape:permuted.clone() });
                        inputs[0] = shapes.len();
                        shapes.push(permuted);
                        static_shape = vec![outer,cols,inner];
                        crate::dispatch::StaticOp::Layout { strides:vec![width,1,cols] }
                    }
                },
                Some(crate::autograd::ForwardOp::SumDim(dim,_)) => {
                    let shape = &shapes[inputs[0]];
                    crate::dispatch::StaticOp::Sum { red:shape[*dim],inner:shape[*dim+1..].iter().product() }
                },
                Some(crate::autograd::ForwardOp::LayerNorm {weight,bias,eps}) => {

                    let shape = &shapes[inputs[0]];
                    let cols = *shape.last().unwrap();
                    crate::dispatch::StaticOp::LayerNorm { rows:shape.iter().product::<usize>()/cols,cols,eps:*eps,weight:*weight,bias:*bias }
                },
                None => {
                    if shapes[inputs[0]] != run.shape {
                        let strides = padded_strides(&shapes[inputs[0]], &run.shape);
                        runs.push(StaticRun { inputs:vec![inputs[0]],op:crate::dispatch::StaticOp::Broadcast { strides },shape:run.shape.clone() });
                        inputs[0] = shapes.len();
                        shapes.push(run.shape.clone());
                    }
                    crate::dispatch::StaticOp::Pointwise(run.steps.clone())
                },
            };
            runs.push(StaticRun { inputs, op, shape: static_shape });
            slots.insert(run.output, shapes.len());
            shapes.push(run.shape.clone());
        }
        if slots[&self.root] != shapes.len()-1 { return Err(unsupported("root is not final static destination")); }
        let mut owners = leaves.clone();
        owners.sort_unstable_by_key(|t| Arc::as_ptr(&t.0.storage));
        owners.dedup_by_key(|t| Arc::as_ptr(&t.0.storage));
        let guards: Vec<_> = owners.iter().map(|t| t.0.storage.write()).collect();
        let buffers = leaves.iter().map(|t| {
            let i = owners.iter().position(|o| Arc::ptr_eq(&o.0.storage, &t.0.storage)).unwrap();
            match &*guards[i] { Storage::Device(b) => b.clone(), _ => unreachable!() }
        }).collect();
        let execution = backend_for(device)?.prepare_static(&runs, buffers)?;
        drop(guards);
        Ok(PreparedChain { execution, leaves: owners, shape: shapes.last().unwrap().clone(), device })
    }

    /// Compile pointwise DAG runs and explicit matmul, bmm, layout, softmax,
    /// sum_dim and LayerNorm forwards. Shared intermediates execute once.
    /// Operations without forward metadata are rejected, never frozen.
    pub fn compile(root: &Tensor) -> Result<CompiledChain> {
        let g = Graph::from_root(root);
        let consumers = g.consumer_counts();
        let mut leaves = HashMap::new();
        for (&id, node) in &g.nodes {
            if node.inputs.is_empty() {
                leaves.insert(id, g.tensors[&id].clone());
            } else if node.tag.is_none() && g.tensors[&id].0.op.as_ref().and_then(|o| o.forward.as_ref()).is_none() {
                return Err(Error::Unsupported {
                    op: "compile_chain", msg: format!("node {id} has no replayable kernel tag"),
                });
            }
        }
        let steps = g.nodes.len() - leaves.len();
        if steps == 0 {
            return Err(Error::Unsupported {
                op: "compile_chain", msg: "root has no recorded operations".into(),
            });
        }
        let metadata = |values: &[usize]| -> Result<Vec<u32>> {
            values.iter().map(|&v| u32::try_from(v).map_err(|_| Error::Unsupported {
                op: "compile_chain", msg: "broadcast metadata exceeds the u32 kernel ABI".into(),
            })).collect()
        };
        let mut used = HashSet::new();
        let mut runs = HashMap::new();
        for &tail in &g.order {
            if leaves.contains_key(&tail) || used.contains(&tail) { continue; }
            if let Some(forward) = g.tensors[&tail].0.op.as_ref().and_then(|o| o.forward.clone()) {
                let node = &g.nodes[&tail];
                runs.insert(tail, CompiledRun { forward: Some(forward), output: tail, inputs: node.inputs.clone(), steps: Vec::new(), shape: node.shape.clone() });
                used.insert(tail);
                continue;
            }
            let mut ids = vec![tail];
            let mut first = tail;
            loop {
                let pred = g.nodes[&first].inputs[0];
                if leaves.contains_key(&pred) || g.nodes[&pred].tag.is_none() || consumers.get(&pred) != Some(&1)
                    || g.nodes[&pred].shape != g.nodes[&first].shape
                    || g.nodes[&pred].inputs.first().is_some_and(|i| g.nodes[i].shape != g.nodes[&pred].shape) {
                    break;
                }
                ids.push(pred);
                first = pred;
            }
            ids.reverse();
            let seed = g.nodes[&first].inputs[0];

            let mut inputs = vec![seed];
            let mut run_steps = Vec::new();
            for id in ids {
                used.insert(id);
                let node = &g.nodes[&id];
                match node.tag.unwrap() {
                    OpTag::Unary(kind) => run_steps.push(ChainStepRef::Unary(kind)),
                    OpTag::Binary(kind) => {
                        let rhs = node.inputs[1];
                        let other = match inputs.iter().position(|&i| i == rhs) {
                            Some(i) => i,
                            None => { inputs.push(rhs); inputs.len() - 1 }
                        };
                        if g.nodes[&rhs].shape == node.shape {
                            run_steps.push(ChainStepRef::Binary { kind, other });
                        } else {
                            run_steps.push(ChainStepRef::BinaryBc {
                                kind, other,
                                dims: metadata(&node.shape)?,
                                strides: metadata(&padded_strides(&g.nodes[&rhs].shape, &node.shape))?,
                            });
                        }
                    }
                }
            }
            runs.insert(tail, CompiledRun {
                forward: None, output: tail, inputs, steps: run_steps, shape: g.nodes[&tail].shape.clone(),
            });
        }
        // A run executes at its tail, after every external dependency's tail.
        let schedule = g.order.iter().rev().filter_map(|id| runs.remove(id)).collect();
        Ok(CompiledChain { leaves, schedule, root: root.id(), steps })
    }

    /// Replay from current leaves without walking the tape or freezing branches.
    /// Each shape-preserving run attempts one resident backend chain launch.
    /// Expanding nodes use ordinary broadcast dispatch, not a fused launch.
    pub fn replay(&self) -> Result<Tensor> {
        crate::capture::without_recording(|| self.replay_inner())
    }

    fn replay_inner(&self) -> Result<Tensor> {
        let mut values = self.leaves.clone();
        for run in &self.schedule {
            let operands: Vec<Tensor> = run.inputs.iter().map(|id| values[id].clone()).collect();
            let out = if let Some(forward) = &run.forward {
                let detached: Vec<_> = operands.iter().map(Tensor::detached_view).collect();
                forward.run(&detached)?
            } else {
                run_chain(&ExecutableChain {
                    steps: run.steps.clone(), operands, out_shape: run.shape.clone(),
                })?
            };
            values.insert(run.output, out);
        }
        Ok(values[&self.root].clone())
    }

    /// Number of distinct captured graph leaves.
    pub fn num_operands(&self) -> usize { self.leaves.len() }

    /// Number of recorded operations, counting shared nodes once.
    pub fn num_steps(&self) -> usize { self.steps }

    /// Number of scheduled runs (backend fallback may issue more launches).
    pub fn num_runs(&self) -> usize { self.schedule.len() }
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
