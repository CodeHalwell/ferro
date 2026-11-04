//! `slice(dim, start, end, step)` and `narrow(dim, start, len)`: zero-copy
//! strided views selecting every `step`-th index of [start, end) along `dim`
//! (offset advances by start*stride, the dim's stride scales by step). `slice`
//! clamps start/end to the dim size like Python slicing; `narrow` is strict.
//! Backward scatters the gradient into a zeros tensor of the input shape on
//! the host (accumulate_grad moves it to the input's device).

use crate::error::{Error, Result};
use crate::shape::default_strides;
use crate::tensor::Tensor;

impl Tensor {
    pub fn slice(&self, dim: usize, start: usize, end: usize, step: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op: "slice", msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        if step == 0 {
            return Err(Error::InvalidShape { op: "slice", msg: "step must be positive".into() });
        }
        let end = end.min(self.0.shape[dim]);
        let start = start.min(self.0.shape[dim]);
        let len = end.saturating_sub(start).div_ceil(step);
        let mut shape = self.0.shape.clone();
        let mut stride = self.0.stride.clone();
        shape[dim] = len;
        let offset = start.checked_mul(stride[dim]).and_then(|value| self.0.offset.checked_add(value))
            .ok_or_else(|| Error::InvalidShape { op: "slice", msg: "slice offset exceeds layout capacity".into() })?;
        stride[dim] = stride[dim].checked_mul(step).ok_or_else(|| Error::InvalidShape { op: "slice", msg: format!("step {step} overflows the stride") })?;
        let view = Tensor::from_parts(self.0.storage.clone(), shape.clone(), stride, offset, self.0.device, false, None);
        let out = self.capture_layout(view)?;
        let in_shape = self.0.shape.clone();
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Slice { dim, start, end, step }, move |g| {
            vec![g.slice_adjoint(&in_shape, dim, start, end, step).unwrap()]
        }))
    }

    /// Linear zero-scatter paired with slice, recorded through the same Op so
    /// successive VJPs alternate between these two actual linear operations.
    pub(crate) fn slice_adjoint(&self, shape: &[usize], dim: usize, start: usize, end: usize, step: usize) -> Result<Tensor> {
        const OP: &str = "slice backward";
        crate::shape::checked_numel(OP, shape)?;
        if dim >= shape.len() || step == 0 || start > shape[dim] || end > shape[dim] {
            return Err(Error::InvalidShape { op: OP, msg: "invalid slice adjoint metadata".into() });
        }
        let mut expected = shape.to_vec();
        expected[dim] = end.saturating_sub(start).div_ceil(step);
        if self.shape() != expected {
            return Err(Error::ShapeMismatch { op: OP, lhs: self.shape().to_vec(), rhs: expected });
        }
        let mut strides = default_strides(shape);
        let base = start * strides[dim];
        strides[dim] = strides[dim].saturating_mul(step);
        macro_rules! scatter {
            ($data:expr, $zero:expr, $ctor:ident) => {{
                let mut values = crate::typed_math::filled(OP, shape, $zero)?;
                let mut index = vec![0usize; shape.len()];
                for value in $data {
                    values[base + index.iter().zip(&strides).map(|(i, s)| i * s).sum::<usize>()] = value;
                    for axis in (0..shape.len()).rev() {
                        index[axis] += 1;
                        if index[axis] < self.shape()[axis] { break; }
                        index[axis] = 0;
                    }
                }
                Tensor::$ctor(values, shape)?
            }};
        }
        let out = match self.dtype() {
            crate::DType::F32 => scatter!(self.to_vec(), 0.0f32, from_vec),
            crate::DType::F64 => scatter!(self.to_vec_f64(), 0.0f64, from_vec_f64),
            dtype => return Err(Error::DtypeMismatch { op: OP, expected: crate::DType::F32, got: dtype }),
        };
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::SliceAdjoint { shape: shape.to_vec(), dim, start, end, step }, move |g| {
            vec![g.slice(dim, start, end, step).unwrap()]
        }))
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op: "narrow", msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        let size = self.0.shape[dim];
        if start.checked_add(len).is_none_or(|e| e > size) {
            return Err(Error::InvalidShape { op: "narrow", msg: format!("range {start}+{len} exceeds dim {dim} of size {size}") });
        }
        self.slice(dim, start, start + len, 1)
    }
}
