//! `permute`: reorder all dims as a zero-copy strided view (shape and stride
//! permuted together, storage shared), the n-dim generalization of transpose.
//! Backward applies the inverse permutation to the gradient, again as a view.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        let ndim = self.ndim();
        let mut seen = vec![false; ndim];
        let valid = dims.len() == ndim && dims.iter().all(|&d| d < ndim && !std::mem::replace(&mut seen[d], true));
        if !valid {
            return Err(Error::InvalidShape { op: "permute", msg: format!("{dims:?} is not a permutation of the {ndim} dims") });
        }
        let out = self.capture_layout(permuted(self, dims))?;
        let mut inv = vec![0usize; ndim];
        for (i, &d) in dims.iter().enumerate() {
            inv[d] = i;
        }
        Ok(out.record_fn(vec![self.clone()], move |g| vec![permuted(g, &inv)]))
    }
}

fn permuted(t: &Tensor, dims: &[usize]) -> Tensor {
    let shape = dims.iter().map(|&d| t.0.shape[d]).collect();
    let stride = dims.iter().map(|&d| t.0.stride[d]).collect();
    Tensor::from_parts(t.0.storage.clone(), shape, stride, t.0.offset, t.0.device, false, None)
}
