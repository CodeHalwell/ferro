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
        let start = start.min(end);
        let len = (end - start).div_ceil(step);
        let mut shape = self.0.shape.clone();
        let mut stride = self.0.stride.clone();
        shape[dim] = len;
        // An empty view addresses nothing; keeping the base offset keeps it
        // inside the buffer for the contiguous slicing fast paths.
        let offset = if len == 0 { self.0.offset } else { self.0.offset + start * stride[dim] };
        stride[dim] = stride[dim].checked_mul(step).ok_or_else(|| Error::InvalidShape { op: "slice", msg: format!("step {step} overflows the stride") })?;
        let view = Tensor::from_parts(self.0.storage.clone(), shape.clone(), stride, offset, self.0.device, false, None);
        let out = self.capture_layout(view)?;
        let in_shape = self.0.shape.clone();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            // Row-major odometer over the output; `base` + idx.strides addresses
            // the matching element of the contiguous input-shaped gradient.
            let mut strides = default_strides(&in_shape);
            let base = start * strides[dim];
            strides[dim] = strides[dim].saturating_mul(step);
            let mut gx = vec![0.0f32; in_shape.iter().product()];
            let mut idx = vec![0usize; shape.len()];
            for v in g.to_vec() {
                gx[base + idx.iter().zip(&strides).map(|(i, s)| i * s).sum::<usize>()] = v;
                for d in (0..shape.len()).rev() {
                    idx[d] += 1;
                    if idx[d] < shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
            vec![Tensor::from_vec(gx, &in_shape).unwrap()]
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
