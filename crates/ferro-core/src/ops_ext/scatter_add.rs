//! `scatter_add`: self[index[i][j][k]][j][k] += src[i][j][k] along `dim`
//! (torch semantics). index and src share a shape no larger than self outside
//! dim. Because additions commute, gradients are exact for any overlap:
//! d/dself = g everywhere, and d/dsrc[k] = g at the scattered position.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn scatter_add(&self, dim: usize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let op = "scatter_add";
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op,
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        if index.dtype() != DType::I64 {
            return Err(Error::DtypeMismatch {
                op,
                expected: DType::I64,
                got: index.dtype(),
            });
        }
        let in_shape = self.shape().to_vec();
        let idx_shape = index.shape().to_vec();
        if idx_shape != src.shape() {
            return Err(Error::ShapeMismatch {
                op,
                lhs: idx_shape.clone(),
                rhs: src.shape().to_vec(),
            });
        }
        if index.ndim() != ndim {
            return Err(Error::InvalidShape {
                op,
                msg: format!("index rank {} must match input rank {ndim}", index.ndim()),
            });
        }
        for d in 0..ndim {
            if d != dim && idx_shape[d] > in_shape[d] {
                return Err(Error::InvalidShape {
                    op,
                    msg: format!(
                        "index shape {idx_shape:?} exceeds input shape {in_shape:?} at dim {d}"
                    ),
                });
            }
        }

        let idx = index.to_vec_i64();
        let s = src.to_vec();
        let dim_size = in_shape[dim];
        let mut coord = vec![0usize; ndim];
        let mut targets = Vec::with_capacity(idx.len());
        for &id in &idx {
            if id < 0 || id as usize >= dim_size {
                return Err(Error::InvalidShape {
                    op,
                    msg: format!("index {id} out of range for dim {dim} with size {dim_size}"),
                });
            }
            let mut off = 0usize;
            for d in 0..ndim {
                let cc = if d == dim { id as usize } else { coord[d] };
                off = off * in_shape[d] + cc;
            }
            targets.push(off);
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if coord[d] < idx_shape[d] {
                    break;
                }
                coord[d] = 0;
            }
        }

        let x = self.to_vec();
        let src_shape = src.shape().to_vec();
        let mut y = x.clone();
        for (t, sv) in targets.iter().zip(&s) {
            y[*t] += sv;
        }
        let out = Tensor::from_vec(y, &in_shape)?;
        if !self.requires_grad() && !src.requires_grad() {
            return Ok(out);
        }

        let src_shape = idx_shape.clone();
        Ok(
            out.record_fn(vec![self.clone(), index.clone(), src.clone()], move |g| {
                let gd = g.to_vec();
                let gsrc: Vec<f32> = targets.iter().map(|&t| gd[t]).collect();
                vec![
                    Tensor::from_vec(gd, &in_shape).unwrap(),
                    Tensor::zeros(&idx_shape),
                    Tensor::from_vec(gsrc, &src_shape).unwrap(),
                ]
            }),
        )
    }
}
