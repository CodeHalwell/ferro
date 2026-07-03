//! `index_select`: pick rows along `dim` by an explicit usize index list.
//! Forward copies contiguous `inner` blocks; backward scatter-adds grad blocks
//! back to the input shape (add, not overwrite, so duplicate indices accumulate).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    /// Tensor-index variant: `indices` must be a 1-D I64 tensor with in-range,
    /// non-negative entries. Delegates to the slice version, so the gradient
    /// to `self` comes from the same recorded backward.
    pub fn index_select_t(&self, dim: usize, indices: &Tensor) -> Result<Tensor> {
        if indices.dtype() != DType::I64 {
            return Err(Error::DtypeMismatch { op: "index_select", expected: DType::I64, got: indices.dtype() });
        }
        if indices.ndim() != 1 {
            return Err(Error::InvalidShape {
                op: "index_select",
                msg: format!("indices must be 1-D, got shape {:?}", indices.shape()),
            });
        }
        let idx: Vec<usize> = indices
            .to_vec_i64()
            .into_iter()
            .map(|i| {
                usize::try_from(i).map_err(|_| Error::InvalidShape {
                    op: "index_select",
                    msg: format!("negative index {i} is not supported"),
                })
            })
            .collect::<Result<_>>()?;
        self.index_select(dim, &idx)
    }

    pub fn index_select(&self, dim: usize, indices: &[usize]) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op: "index_select", msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        let in_shape = self.shape().to_vec();
        let dim_size = in_shape[dim];
        if let Some(&bad) = indices.iter().find(|&&i| i >= dim_size) {
            return Err(Error::InvalidShape { op: "index_select", msg: format!("index {bad} out of range for dim {dim} with size {dim_size}") });
        }

        let inner: usize = in_shape[dim + 1..].iter().product();
        let outer: usize = in_shape[..dim].iter().product();
        let data = self.to_vec();
        let mut out_data = Vec::with_capacity(outer * indices.len() * inner);
        for o in 0..outer {
            for &idx in indices {
                let start = (o * dim_size + idx) * inner;
                out_data.extend_from_slice(&data[start..start + inner]);
            }
        }
        let mut out_shape = in_shape.clone();
        out_shape[dim] = indices.len();
        let out = Tensor::from_vec(out_data, &out_shape)?;

        let indices = indices.to_vec();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let mut gx = vec![0.0f32; in_shape.iter().product()];
            let mut src = 0;
            for o in 0..outer {
                for &idx in &indices {
                    let dst = (o * dim_size + idx) * inner;
                    for j in 0..inner {
                        gx[dst + j] += gd[src + j];
                    }
                    src += inner;
                }
            }
            vec![Tensor::from_vec(gx, &in_shape).unwrap()]
        }))
    }
}
