//! `index_add(dim, index, source)`: out = self with
//! out[.., index[i], ..] += source[.., i, ..] along `dim` (torch semantics,
//! alpha = 1). `index` is a 1-D I64 tensor; duplicates accumulate. Gradients:
//! d/dself = g, d/dsource = g gathered at the indexed rows. Host compute.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn index_add(&self, dim: usize, index: &Tensor, source: &Tensor) -> Result<Tensor> {
        let op = "index_add";
        for t in [self, source] {
            if t.dtype() != DType::F32 {
                return Err(Error::DtypeMismatch { op, expected: DType::F32, got: t.dtype() });
            }
        }
        if index.dtype() != DType::I64 {
            return Err(Error::DtypeMismatch { op, expected: DType::I64, got: index.dtype() });
        }
        for t in [index, source] {
            if t.device() != self.device() {
                return Err(Error::DeviceMismatch { op, lhs: self.device(), rhs: t.device() });
            }
        }
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op, msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        if index.ndim() != 1 {
            return Err(Error::InvalidShape { op, msg: format!("index must be 1-D, got shape {:?}", index.shape()) });
        }
        let shape = self.shape().to_vec();
        let mut want = shape.clone();
        want[dim] = index.numel();
        if source.shape() != want {
            return Err(Error::ShapeMismatch { op, lhs: want, rhs: source.shape().to_vec() });
        }
        let size = shape[dim];
        let idx: Vec<usize> = index.to_vec_i64().into_iter().map(|i| {
            usize::try_from(i).ok().filter(|&u| u < size)
                .ok_or_else(|| Error::InvalidShape { op, msg: format!("index {i} out of range for dim {dim} with size {size}") })
        }).collect::<Result<_>>()?;

        let inner: usize = shape[dim + 1..].iter().product();
        let outer: usize = shape[..dim].iter().product();
        let mut y = self.to_vec();
        let src = source.to_vec();
        let mut s = 0;
        for o in 0..outer {
            for &ix in &idx {
                let dst = (o * size + ix) * inner;
                for j in 0..inner {
                    y[dst + j] += src[s + j];
                }
                s += inner;
            }
        }
        let out = Tensor::from_vec(y, &shape)?;
        Ok(out.record_fn(vec![self.clone(), source.clone()], move |g| {
            vec![g.clone(), g.index_select(dim, &idx).unwrap()]
        }))
    }
}
