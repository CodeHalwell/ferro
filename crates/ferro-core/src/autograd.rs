use std::collections::HashSet;

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
    /// Reverse-mode autodiff seeded with ones (call on a scalar loss). Populates
    /// `.grad()` on every leaf/intermediate with `requires_grad = true`. Repeated
    /// calls follow torch's retain_graph semantics: leaf grads accumulate across
    /// calls, interior grads are recomputed from scratch each call.
    pub fn backward(&self) {
        assert!(
            self.numel() == 1,
            "backward() requires a scalar output (single element), got shape {:?}; \
             reduce with .sum() or .mean() first",
            self.shape()
        );
        let topo = build_topo(self);
        // Interior grads are scratch state from any prior backward call; left in
        // place they would compound through accumulate_grad and corrupt results.
        for t in &topo {
            if t.0.op.is_some() {
                t.zero_grad();
            }
        }
        let seed = Tensor::full_on(self.shape(), 1.0, self.device())
            .expect("loss tensor's device backend is registered");
        self.set_grad(seed);
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
