use std::collections::HashSet;

use crate::shape::numel;
use crate::tensor::{raw_binary, raw_matmul, raw_unary, unbroadcast, Tensor};

/// The op that produced a tensor, holding just enough to run its vector-Jacobian
/// product in reverse mode. Ops that need their own output for the backward pass
/// (exp, sigmoid) stash a *detached* snapshot to avoid a reference cycle.
pub(crate) enum Op {
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),
    Matmul(Tensor, Tensor),
    Sum(Tensor, Vec<usize>),
    Mean(Tensor, Vec<usize>),
    Relu(Tensor),
    Exp(Tensor, Tensor),
    Sigmoid(Tensor, Tensor),
    Neg(Tensor),
    Reshape(Tensor, Vec<usize>),
    Transpose(Tensor, usize, usize),
    /// Extensibility escape hatch: a self-contained op carrying its own inputs
    /// and vector-Jacobian-product closure (grads returned in input order). New
    /// ops use this via `Tensor::record_fn` and live in their own files instead
    /// of growing this enum.
    Fn(Vec<Tensor>, Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>),
}

impl Op {
    pub(crate) fn inputs(&self) -> Vec<&Tensor> {
        match self {
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) | Op::Matmul(a, b) => {
                vec![a, b]
            }
            Op::Sum(a, _)
            | Op::Mean(a, _)
            | Op::Relu(a)
            | Op::Exp(a, _)
            | Op::Sigmoid(a, _)
            | Op::Neg(a)
            | Op::Reshape(a, _)
            | Op::Transpose(a, _, _) => vec![a],
            Op::Fn(inputs, _) => inputs.iter().collect(),
        }
    }

    /// Consume the op, yielding every tensor it holds (inputs plus any saved
    /// snapshots). Used by TensorInner's iterative Drop to unlink deep graphs
    /// without recursing.
    pub(crate) fn into_inputs(self) -> Vec<Tensor> {
        match self {
            Op::Add(a, b)
            | Op::Sub(a, b)
            | Op::Mul(a, b)
            | Op::Div(a, b)
            | Op::Matmul(a, b)
            | Op::Exp(a, b)
            | Op::Sigmoid(a, b) => vec![a, b],
            Op::Sum(a, _)
            | Op::Mean(a, _)
            | Op::Relu(a)
            | Op::Neg(a)
            | Op::Reshape(a, _)
            | Op::Transpose(a, _, _) => vec![a],
            Op::Fn(inputs, _) => inputs,
        }
    }

    /// Given the gradient flowing into this op's output, return the gradient for
    /// each input (in `inputs()` order).
    pub(crate) fn backward(&self, g: &Tensor) -> Vec<Tensor> {
        match self {
            Op::Add(a, b) => vec![unbroadcast(g, a.shape()), unbroadcast(g, b.shape())],
            Op::Sub(a, b) => {
                let neg = raw_unary(g, |x| -x);
                vec![unbroadcast(g, a.shape()), unbroadcast(&neg, b.shape())]
            }
            Op::Mul(a, b) => {
                let ga = raw_binary("mul_bw", g, b, |x, y| x * y).unwrap();
                let gb = raw_binary("mul_bw", g, a, |x, y| x * y).unwrap();
                vec![unbroadcast(&ga, a.shape()), unbroadcast(&gb, b.shape())]
            }
            Op::Div(a, b) => {
                let ga = raw_binary("div_bw", g, b, |x, y| x / y).unwrap();
                let ga_num = raw_binary("div_bw", g, a, |x, y| x * y).unwrap();
                let gb = raw_binary("div_bw", &ga_num, b, |x, y| -x / (y * y)).unwrap();
                vec![unbroadcast(&ga, a.shape()), unbroadcast(&gb, b.shape())]
            }
            Op::Matmul(a, b) => {
                // C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC
                let da = raw_matmul(g, &b.transpose_view(0, 1).unwrap()).unwrap();
                let db = raw_matmul(&a.transpose_view(0, 1).unwrap(), g).unwrap();
                vec![da, db]
            }
            Op::Sum(_, in_shape) => vec![Tensor::full(in_shape, g.item())],
            Op::Mean(_, in_shape) => {
                vec![Tensor::full(in_shape, g.item() / numel(in_shape) as f32)]
            }
            Op::Relu(a) => {
                let mask = raw_binary("relu_bw", g, a, |gg, aa| if aa > 0.0 { gg } else { 0.0 });
                vec![mask.unwrap()]
            }
            Op::Exp(_, out) => vec![raw_binary("exp_bw", g, out, |x, y| x * y).unwrap()],
            Op::Sigmoid(_, out) => {
                vec![raw_binary("sigmoid_bw", g, out, |x, y| x * y * (1.0 - y)).unwrap()]
            }
            Op::Neg(_) => vec![raw_unary(g, |x| -x)],
            Op::Reshape(_, in_shape) => vec![Tensor::from_vec(g.to_vec(), in_shape).unwrap()],
            Op::Transpose(_, d0, d1) => vec![g.transpose_view(*d0, *d1).unwrap()],
            Op::Fn(_, f) => f(g),
        }
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
        self.set_grad(Tensor::ones(self.shape()));
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
            for (inp, ig) in inputs.into_iter().zip(grads) {
                if inp.requires_grad() {
                    inp.accumulate_grad(ig);
                }
            }
        }
    }
}
