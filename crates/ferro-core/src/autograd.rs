use std::collections::HashSet;

use crate::dtype::DType;
use crate::tensor::Tensor;

/// The graph node behind every autograd-recorded tensor: the op's inputs plus a
/// vector-Jacobian-product closure returning one gradient per input, in order.
/// Ops that need their own output for backward (exp, sigmoid, softmax) capture
/// a *detached* snapshot in the closure to avoid a reference cycle.
pub(crate) struct Op {
    inputs: Vec<Tensor>,
    backward: Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
}

impl Op {
    pub(crate) fn new(
        inputs: Vec<Tensor>,
        backward: Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
    ) -> Op {
        Op { inputs, backward }
    }

    pub(crate) fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }

    pub(crate) fn backward(&self, g: &Tensor) -> Vec<Tensor> {
        (self.backward)(g)
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
    /// pass itself (create_graph) is a separate, later capability - see
    /// docs/CAPABILITY.md 1.2. Same retain_graph semantics as `backward()`.
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
            let g = t.grad().expect("every op node on the path receives a gradient");
            let inputs = op.inputs();
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
