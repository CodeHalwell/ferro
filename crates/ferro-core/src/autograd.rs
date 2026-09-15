use std::collections::{HashMap, HashSet};
use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Whether ordinary operations record autograd on this thread. Capture is independent.
pub fn is_grad_enabled() -> bool { GRAD_ENABLED.with(Cell::get) }

/// Set this thread's recording mode and return the previous mode for restoration.
pub fn set_grad_enabled(enabled: bool) -> bool { GRAD_ENABLED.with(|mode| mode.replace(enabled)) }

struct GradModeGuard(bool);
impl Drop for GradModeGuard {
    fn drop(&mut self) { set_grad_enabled(self.0); }
}

/// Run with a recording mode, restoring the previous mode even during unwinding.
pub fn with_grad_enabled<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    let _guard = GradModeGuard(set_grad_enabled(enabled));
    run()
}

#[path = "higher_order.rs"]
mod higher_order;

use crate::dispatch::OpTag;
use crate::dtype::DType;
use crate::tensor::Tensor;

/// Forward executors use current operands, never autograd backward closures.
#[derive(Clone, Debug)]
pub(crate) enum ForwardOp {
    Sum, Mean, MatMul, Bmm, Reshape(Vec<usize>), Transpose(usize, usize),
    Softmax(usize), SumDim(usize, bool), LayerNorm { weight: bool, bias: bool, eps: f32 },
}

impl ForwardOp {
    pub(crate) fn run(&self, inputs: &[Tensor]) -> crate::Result<Tensor> {
        match self {
            Self::Sum => Ok(inputs[0].sum()),
            Self::Mean => Ok(inputs[0].mean()),
            Self::MatMul => inputs[0].matmul(&inputs[1]),
            Self::Bmm => inputs[0].bmm(&inputs[1]),
            Self::Reshape(shape) => inputs[0].reshape(shape),
            Self::Transpose(a, b) => inputs[0].transpose(*a, *b),
            Self::Softmax(dim) => inputs[0].softmax(*dim),
            Self::SumDim(dim, keep) => inputs[0].sum_dim(*dim, *keep),
            Self::LayerNorm { weight, bias, eps } => inputs[0].layer_norm(weight.then_some(&inputs[1]), bias.then_some(&inputs[2]), *eps),
        }
    }
}

/// The graph node behind every autograd-recorded tensor: the op's inputs plus a
/// vector-Jacobian-product closure returning one gradient per input, in order.
/// Ops that need their own output for backward (exp, sigmoid, softmax) capture
/// a *detached* snapshot in the closure to avoid a reference cycle - detaching
/// allocates fresh storage, so those closures are immune by construction to
/// the version check below (mutating the live output cannot poison a copy
/// that does not share its storage).
///
/// `saved_versions` snapshots each input's storage version (`Tensor::version`)
/// at record time; `Tensor::backward` asserts they are unchanged immediately
/// before running this op's closure, turning a mutation of a saved input
/// between forward and backward into a loud error instead of a silently wrong
/// gradient.
pub(crate) struct Op {
    pub(crate) forward: Option<ForwardOp>,
    inputs: Vec<Tensor>,
    saved_versions: Vec<u64>,
    /// Which named kernel this op ran (kind-routed ops only); None for
    /// composite ops. Read by the graph compiler to plan fused execution.
    pub(crate) tag: Option<OpTag>,
    backward: Option<Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>>,
}

impl Op {
    pub(crate) fn new(
        inputs: Vec<Tensor>,
        backward: Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
    ) -> Op {
        let saved_versions = inputs.iter().map(|t| t.version()).collect();
        Op {
            inputs,
            saved_versions,
            forward: None,
            tag: None,
            backward: Some(backward),
        }
    }

    /// Like `new` but carrying the kernel tag for fusion planning.
    pub(crate) fn new_tagged(
        inputs: Vec<Tensor>,
        tag: OpTag,
        backward: Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
    ) -> Op {
        let saved_versions = inputs.iter().map(|t| t.version()).collect();
        Op {
            inputs,
            saved_versions,
            forward: None,
            tag: Some(tag),
            backward: Some(backward),
        }
    }

    pub(crate) fn inference(inputs: Vec<Tensor>, tag: Option<OpTag>) -> Op {
        Op { inputs, saved_versions: Vec::new(), tag, forward: None, backward: None }
    }

    pub(crate) fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }

    pub(crate) fn saved_versions(&self) -> &[u64] {
        &self.saved_versions
    }

    pub(crate) fn backward(&self, g: &Tensor) -> Vec<Tensor> {
        self.vjp(g, false).expect("inference-only node has no backward")
    }

    fn vjp(&self, g: &Tensor, create_graph: bool) -> crate::Result<Vec<Tensor>> {
        let backward = self.backward.as_ref().ok_or_else(|| crate::Error::Unsupported {
            op: "vjp_wrt", msg: "inference-only node has no backward".into(),
        })?;
        if create_graph { self.graph_backward(g) } else { Ok(backward(g)) }
    }

    /// Consume the op, yielding its input tensors. Used by TensorInner's
    /// iterative Drop to unlink deep graphs without recursing.
    pub(crate) fn into_inputs(self) -> Vec<Tensor> {
        self.inputs
    }
}

/// Post-order topological sort via an explicit stack of (node, next child
/// index), so graphs deeper than the native stack cannot overflow it.
fn build_topo(root: &Tensor) -> Vec<Tensor> {
    let mut seen = HashSet::new();
    let mut topo = Vec::new();
    let mut stack: Vec<(Tensor, usize)> = Vec::new();
    seen.insert(root.id());
    stack.push((root.clone(), 0));
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
        topo.push(t);
    }
    topo
}

impl Tensor {
    /// Functional scalar gradient in input order, without touching `.grad()`.
    /// Unused/non-requiring inputs return detached zeros, which can be differentiated again.
    /// create_graph supports add/sub/mul/div/neg/exp/sigmoid/tanh/sqrt, 2D matmul,
    /// reshape/transpose/sum_dim/sum/mean. Other VJPs return Error::Unsupported, never
    /// silently detached derivatives. Broadcasting and accumulation retain graphs.
    pub fn grad_wrt(&self, inputs: &[&Tensor], create_graph: bool) -> crate::Result<Vec<Tensor>> {
        if self.numel() != 1 {
            return Err(crate::Error::InvalidShape { op: "grad_wrt", msg: "requires a single-element output; use vjp_wrt with an explicit seed".into() });
        }
        let seed = Tensor::full_on(self.shape(), 1.0, self.device())?;
        self.vjp_wrt(inputs, &seed, create_graph)
    }

    /// Functional VJP for selected leaves or intermediates. With create_graph,
    /// the seed is a live differentiable operand (held fixed in this VJP but
    /// available to later derivatives). Otherwise it is detached. Graphs are
    /// retained; returned derivatives never accumulate into `.grad()`.
    /// create_graph controls recording during this VJP even inside no-grad; the
    /// caller's thread-local mode is restored on return, error, or unwinding.
    pub fn vjp_wrt(&self, inputs: &[&Tensor], seed: &Tensor, create_graph: bool) -> crate::Result<Vec<Tensor>> {
        let _grad_mode = GradModeGuard(set_grad_enabled(create_graph));
        use crate::Error;
        if seed.shape() != self.shape() {
            return Err(Error::ShapeMismatch { op: "vjp_wrt", lhs: self.shape().to_vec(), rhs: seed.shape().to_vec() });
        }
        for t in std::iter::once(self).chain(std::iter::once(seed)).chain(inputs.iter().copied()) {
            if t.dtype() != DType::F32 {
                return Err(Error::DtypeMismatch { op: "vjp_wrt", expected: DType::F32, got: t.dtype() });
            }
            if t.device() != self.device() {
                return Err(Error::DeviceMismatch { op: "vjp_wrt", lhs: self.device(), rhs: t.device() });
            }
        }
        let topo = build_topo(self);
        let selected: HashSet<_> = inputs.iter().filter(|t| t.requires_grad()).map(|t| t.id()).collect();
        let mut needed = selected;
        for t in &topo {
            if let Some(op) = &t.0.op {
                if op.inputs.iter().any(|i| needed.contains(&i.id())) { needed.insert(t.id()); }
            }
        }
        let mut grads = HashMap::new();
        if needed.contains(&self.id()) {
            grads.insert(self.id(), if create_graph { seed.clone() } else { seed.detach_copy() });
        }
        for t in topo.iter().rev() {
            let Some(op) = &t.0.op else { continue };
            if !op.inputs.iter().any(|i| needed.contains(&i.id()) && i.requires_grad()) { continue; }
            let Some(g) = grads.get(&t.id()) else { continue };
            for (inp, saved) in op.inputs.iter().zip(&op.saved_versions) {
                if inp.version() != *saved {
                    return Err(Error::Unsupported { op: "vjp_wrt", msg: format!("saved input {} modified inplace (version {} -> {})", inp.id(), saved, inp.version()) });
                }
            }
            let contributions = op.vjp(g, create_graph)?;
            if contributions.len() != op.inputs.len() {
                return Err(Error::Unsupported { op: "vjp_wrt", msg: "backward gradient arity does not match inputs".into() });
            }
            for (inp, contribution) in op.inputs.iter().zip(contributions) {
                if contribution.shape() != inp.shape() {
                    return Err(Error::ShapeMismatch { op: "vjp_wrt backward", lhs: inp.shape().to_vec(), rhs: contribution.shape().to_vec() });
                }
                if !inp.requires_grad() || !needed.contains(&inp.id()) { continue; }
                let contribution = if contribution.device() != inp.device() {
                    if create_graph { return Err(Error::DeviceMismatch { op: "create_graph backward", lhs: inp.device(), rhs: contribution.device() }); }
                    contribution.to_device(inp.device())?
                } else { contribution };
                let value = if let Some(previous) = grads.remove(&inp.id()) { previous.add(&contribution)? } else { contribution };
                grads.insert(inp.id(), if create_graph { value } else { value.detach_copy() });
            }
        }
        inputs.iter().map(|input| {
            match grads.get(&input.id()) {
                Some(g) => Ok(if create_graph { g.clone() } else { g.detach_copy() }),
                None => Tensor::full_on(input.shape(), 0.0, input.device()),
            }
        }).collect()
    }

    /// Reverse-mode autodiff seeded with ones (call on a scalar loss): the
    /// scalar restriction (v = 1) of `backward_with`. Populates `.grad()` on
    /// every leaf/intermediate with `requires_grad = true`. Repeated calls
    /// follow torch's retain_graph semantics: leaf grads accumulate across
    /// calls, interior grads are recomputed from scratch each call.
    pub fn backward(&self) {
        assert!(
            self.numel() == 1,
            "backward() requires a scalar output (single element), got shape {:?}; \
             reduce with .sum() or .mean() first",
            self.shape()
        );
        let seed = Tensor::full_on(self.shape(), 1.0, self.device())
            .expect("loss tensor's device backend is registered");
        self.backward_with(&seed);
    }

    /// Reverse-mode autodiff seeded with an explicit cotangent `v`: computes
    /// the vector-Jacobian product v^T J for any (not just scalar) root, so
    /// `backward()` is just the v = 1 case on a scalar. `cotangent` must match
    /// `self`'s shape and be f32; it is detached before seeding (a graph-
    /// carrying cotangent must not splice a surprise edge into this backward
    /// pass) and aligned to `self`'s device the same way any op's backward
    /// contribution is, via `accumulate_grad`. Differentiating through this
    /// pass itself uses the functional `grad_wrt`/`vjp_wrt` APIs instead.
    /// Same retain_graph semantics as `backward()`.
    pub fn backward_with(&self, cotangent: &Tensor) {
        assert!(
            cotangent.shape() == self.shape(),
            "backward_with() cotangent shape {:?} does not match output shape {:?}",
            cotangent.shape(),
            self.shape()
        );
        assert!(
            cotangent.dtype() == DType::F32,
            "backward_with() cotangent must be f32 (autograd is f32-only), got {}",
            cotangent.dtype()
        );
        let topo = build_topo(self);
        // Interior grads are scratch state from any prior backward call; left in
        // place they would compound through accumulate_grad and corrupt results.
        for t in &topo {
            if t.0.op.is_some() {
                t.zero_grad();
            }
        }
        // Accumulate rather than set: a leaf root (no op node) keeps its grad
        // across backward calls like any other leaf; op roots were cleared above.
        self.accumulate_grad(cotangent.detach_copy());
        for t in topo.iter().rev() {
            let Some(op) = &t.0.op else { continue };
            if op.backward.is_none() { continue; }
            let g = t
                .grad()
                .expect("every op node on the path receives a gradient");
            let inputs = op.inputs();
            for (i, (inp, &saved)) in inputs.iter().zip(op.saved_versions()).enumerate() {
                let now = inp.version();
                assert!(
                    now == saved,
                    "one of the variables needed for gradient computation has been modified by \
                     an inplace operation: input {i} of the op producing tensor id {} was saved \
                     with version {saved} but is now version {now}",
                    t.id()
                );
            }
            let grads = op.backward(&g);
            assert!(
                grads.len() == inputs.len(),
                "op backward returned {} gradients for {} inputs; record_fn closures \
                 must return one gradient per input",
                grads.len(),
                inputs.len()
            );
            for (inp, ig) in inputs.iter().zip(grads) {
                if inp.requires_grad() {
                    inp.accumulate_grad(ig);
                }
            }
        }
    }
}
