use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::autograd::Op;
use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::shape::{broadcast_shapes, default_strides, numel};

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

fn fresh_id() -> usize {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// MVP storage is f32-only, but kept behind an enum so more dtypes slot in
/// without touching the tensor/view/autograd machinery.
#[derive(Debug)]
pub enum Storage {
    F32(Vec<f32>),
}

impl Storage {
    pub fn as_f32(&self) -> &[f32] {
        match self {
            Storage::F32(v) => v,
        }
    }
}

pub(crate) struct TensorInner {
    pub(crate) id: usize,
    pub(crate) storage: Arc<Storage>,
    pub(crate) shape: Vec<usize>,
    pub(crate) stride: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) requires_grad: bool,
    /// How this tensor was produced, for reverse-mode autodiff. `None` for leaves.
    pub(crate) op: Option<Op>,
    pub(crate) grad: Mutex<Option<Tensor>>,
}

/// Reference-counted, autograd-aware tensor. Cloning is cheap (bumps an `Arc`)
/// and shares identity, so a value used in several ops accumulates grad once.
#[derive(Clone)]
pub struct Tensor(pub(crate) Arc<TensorInner>);

impl Tensor {
    pub(crate) fn from_parts(
        storage: Arc<Storage>,
        shape: Vec<usize>,
        stride: Vec<usize>,
        offset: usize,
        requires_grad: bool,
        op: Option<Op>,
    ) -> Tensor {
        Tensor(Arc::new(TensorInner {
            id: fresh_id(),
            storage,
            shape,
            stride,
            offset,
            requires_grad,
            op,
            grad: Mutex::new(None),
        }))
    }

    /// Build a contiguous leaf tensor from row-major data.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        if data.len() != numel(shape) {
            return Err(Error::InvalidShape {
                op: "from_vec",
                msg: format!("{} elements do not fit shape {shape:?}", data.len()),
            });
        }
        Ok(Tensor::from_parts(
            Arc::new(Storage::F32(data)),
            shape.to_vec(),
            default_strides(shape),
            0,
            false,
            None,
        ))
    }

    pub fn full(shape: &[usize], value: f32) -> Tensor {
        Tensor::from_vec(vec![value; numel(shape)], shape).unwrap()
    }

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 0.0)
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::full(shape, 1.0)
    }

    pub fn scalar(value: f32) -> Tensor {
        Tensor::from_vec(vec![value], &[]).unwrap()
    }

    pub fn randn(shape: &[usize], rng: &Rng) -> Tensor {
        let data = (0..numel(shape)).map(|_| rng.normal()).collect();
        Tensor::from_vec(data, shape).unwrap()
    }

    // --- metadata ---------------------------------------------------------

    pub fn id(&self) -> usize {
        self.0.id
    }
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }
    pub fn ndim(&self) -> usize {
        self.0.shape.len()
    }
    pub fn numel(&self) -> usize {
        numel(&self.0.shape)
    }
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad
    }

    /// Mark a leaf as requiring gradients (like `tensor.requires_grad_(True)`).
    /// Returns a fresh leaf sharing storage; only meaningful on leaves.
    pub fn requires_grad_(&self, req: bool) -> Tensor {
        Tensor::from_parts(
            self.0.storage.clone(),
            self.0.shape.clone(),
            self.0.stride.clone(),
            self.0.offset,
            req,
            None,
        )
    }

    // --- materialization --------------------------------------------------

    /// Gather this (possibly strided/broadcast) view into a contiguous
    /// row-major Vec. Every compute kernel reads through this, so strided
    /// views (transpose, broadcast) work transparently.
    pub fn to_vec(&self) -> Vec<f32> {
        let inner = &self.0;
        let data = inner.storage.as_f32();
        let n = self.numel();
        let ndim = inner.shape.len();
        if ndim == 0 {
            return vec![data[inner.offset]];
        }
        let mut out = Vec::with_capacity(n);
        let mut idx = vec![0usize; ndim];
        for _ in 0..n {
            let mut off = inner.offset;
            for d in 0..ndim {
                off += idx[d] * inner.stride[d];
            }
            out.push(data[off]);
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < inner.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    }

    /// Scalar value of a 0-d (or single-element) tensor.
    pub fn item(&self) -> f32 {
        self.to_vec()[0]
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        self.0.stride == default_strides(&self.0.shape)
    }

    // --- grad storage -----------------------------------------------------

    pub fn grad(&self) -> Option<Tensor> {
        self.0.grad.lock().unwrap().clone()
    }

    pub fn zero_grad(&self) {
        *self.0.grad.lock().unwrap() = None;
    }

    pub(crate) fn set_grad(&self, g: Tensor) {
        *self.0.grad.lock().unwrap() = Some(g);
    }

    pub(crate) fn accumulate_grad(&self, g: Tensor) {
        let mut slot = self.0.grad.lock().unwrap();
        *slot = Some(match slot.take() {
            None => g,
            Some(existing) => raw_binary("grad_acc", &existing, &g, |a, b| a + b).unwrap(),
        });
    }

    // --- views (share storage) -------------------------------------------

    /// Broadcast to `shape` without copying (inserts zero strides). Detached:
    /// broadcasting's gradient is handled by reducing in backward.
    pub(crate) fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        let cur = &self.0.shape;
        if shape.len() < cur.len() {
            return Err(Error::ShapeMismatch { op: "broadcast_to", lhs: cur.clone(), rhs: shape.to_vec() });
        }
        let pad = shape.len() - cur.len();
        let mut new_stride = vec![0usize; shape.len()];
        for i in 0..shape.len() {
            if i < pad {
                new_stride[i] = 0;
            } else {
                let ci = i - pad;
                if cur[ci] == shape[i] {
                    new_stride[i] = self.0.stride[ci];
                } else if cur[ci] == 1 {
                    new_stride[i] = 0;
                } else {
                    return Err(Error::ShapeMismatch { op: "broadcast_to", lhs: cur.clone(), rhs: shape.to_vec() });
                }
            }
        }
        Ok(Tensor::from_parts(
            self.0.storage.clone(),
            shape.to_vec(),
            new_stride,
            self.0.offset,
            false,
            None,
        ))
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        if numel(shape) != self.numel() {
            return Err(Error::InvalidShape {
                op: "reshape",
                msg: format!("cannot reshape {:?} into {shape:?}", self.0.shape),
            });
        }
        // reshape needs contiguous data; materialize if this is a strided view.
        let base = if self.is_contiguous() {
            self.clone()
        } else {
            Tensor::from_vec(self.to_vec(), &self.0.shape)?
        };
        let out = Tensor::from_parts(
            base.0.storage.clone(),
            shape.to_vec(),
            default_strides(shape),
            base.0.offset,
            self.0.requires_grad,
            None,
        );
        Ok(out.record(self.0.requires_grad, || Op::Reshape(self.clone(), self.0.shape.clone())))
    }

    /// Detached transpose view (swaps two dims' shape/stride, shares storage).
    pub(crate) fn transpose_view(&self, d0: usize, d1: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if d0 >= ndim || d1 >= ndim {
            return Err(Error::InvalidShape {
                op: "transpose",
                msg: format!("dims ({d0},{d1}) out of range for rank {ndim}"),
            });
        }
        let mut shape = self.0.shape.clone();
        let mut stride = self.0.stride.clone();
        shape.swap(d0, d1);
        stride.swap(d0, d1);
        Ok(Tensor::from_parts(self.0.storage.clone(), shape, stride, self.0.offset, false, None))
    }

    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Tensor> {
        let out = self.transpose_view(d0, d1)?;
        Ok(out.record(self.0.requires_grad, || Op::Transpose(self.clone(), d0, d1)))
    }

    /// A detached, contiguous copy that shares no autograd history or storage.
    pub fn detach_copy(&self) -> Tensor {
        Tensor::from_vec(self.to_vec(), &self.0.shape).unwrap()
    }

    /// Attach an autograd op to a freshly-computed output when grad is needed.
    pub(crate) fn record(mut self, requires_grad: bool, op: impl FnOnce() -> Op) -> Tensor {
        if requires_grad {
            let inner = Arc::get_mut(&mut self.0).expect("fresh output is uniquely owned");
            inner.requires_grad = true;
            inner.op = Some(op());
        }
        self
    }

    /// Extensible autograd hook for ops defined outside the core `Op` enum.
    /// `inputs` are the differentiable operands; `backward` maps the output
    /// gradient to one gradient per input (same order). Recorded only when some
    /// input requires grad. `self` must be a freshly-created, uniquely-owned
    /// output (as returned by the raw kernels).
    pub fn record_fn<F>(mut self, inputs: Vec<Tensor>, backward: F) -> Tensor
    where
        F: Fn(&Tensor) -> Vec<Tensor> + Send + Sync + 'static,
    {
        if inputs.iter().any(|t| t.requires_grad()) {
            let inner = Arc::get_mut(&mut self.0).expect("fresh output is uniquely owned");
            inner.requires_grad = true;
            inner.op = Some(Op::Fn(inputs, Box::new(backward)));
        }
        self
    }
}

// --- raw (detached) compute kernels --------------------------------------
// These never record autograd; forward wrappers and backward both call them.

pub(crate) fn raw_binary(
    op: &'static str,
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f32, f32) -> f32,
) -> Result<Tensor> {
    let out_shape = broadcast_shapes(op, &a.0.shape, &b.0.shape)?;
    let va = a.broadcast_to(&out_shape)?.to_vec();
    let vb = b.broadcast_to(&out_shape)?.to_vec();
    let data = va.iter().zip(vb.iter()).map(|(&x, &y)| f(x, y)).collect();
    Tensor::from_vec(data, &out_shape)
}

pub(crate) fn raw_unary(a: &Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    let data = a.to_vec().into_iter().map(f).collect();
    Tensor::from_vec(data, &a.0.shape).unwrap()
}

/// 2-D matmul: (m,k) @ (k,n) -> (m,n). Higher ranks are a follow-up.
pub(crate) fn raw_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(Error::Unsupported {
            op: "matmul",
            msg: format!("only 2-D supported in MVP, got {:?} and {:?}", a.0.shape, b.0.shape),
        });
    }
    let (m, k) = (a.0.shape[0], a.0.shape[1]);
    let (k2, n) = (b.0.shape[0], b.0.shape[1]);
    if k != k2 {
        return Err(Error::ShapeMismatch { op: "matmul", lhs: a.0.shape.clone(), rhs: b.0.shape.clone() });
    }
    let va = a.to_vec();
    let vb = b.to_vec();
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = va[i * k + p];
            if aip == 0.0 {
                continue;
            }
            let brow = p * n;
            let orow = i * n;
            for j in 0..n {
                out[orow + j] += aip * vb[brow + j];
            }
        }
    }
    Tensor::from_vec(out, &[m, n])
}

/// Sum over one dim, matching PyTorch's keepdim semantics.
pub(crate) fn raw_sum_dim(t: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    let in_shape = t.0.shape.clone();
    let ndim = in_shape.len();
    let v = t.to_vec();
    let mut keep_shape = in_shape.clone();
    keep_shape[dim] = 1;
    let keep_strides = default_strides(&keep_shape);
    let mut out = vec![0f32; numel(&keep_shape)];
    let mut idx = vec![0usize; ndim];
    for &val in v.iter() {
        let mut off = 0usize;
        for d in 0..ndim {
            let id = if d == dim { 0 } else { idx[d] };
            off += id * keep_strides[d];
        }
        out[off] += val;
        for d in (0..ndim).rev() {
            idx[d] += 1;
            if idx[d] < in_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    let out_shape: Vec<usize> = if keepdim {
        keep_shape
    } else {
        in_shape.iter().enumerate().filter(|(d, _)| *d != dim).map(|(_, &s)| s).collect()
    };
    Tensor::from_vec(out, &out_shape).unwrap()
}

/// Reduce a (broadcasted) gradient back down to `target` shape by summing over
/// the dims that were expanded during the forward broadcast.
pub(crate) fn unbroadcast(g: &Tensor, target: &[usize]) -> Tensor {
    let mut g = g.clone();
    while g.ndim() > target.len() {
        g = raw_sum_dim(&g, 0, false);
    }
    for d in 0..target.len() {
        if target[d] == 1 && g.0.shape[d] != 1 {
            g = raw_sum_dim(&g, d, true);
        }
    }
    g
}
