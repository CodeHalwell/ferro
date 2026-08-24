//! `scatter`: out[index[i][j][k]][j][k] = src[i][j][k] along `dim` (torch
//! semantics: write, not add - distinct from `scatter_add`). Positions never
//! written keep self's value; duplicate indices overwrite left-to-right, so
//! the last writer wins. That deterministic ordering makes the backward
//! exact even under duplicates: d/dsrc goes to each target's last writer
//! only (earlier writers contributed nothing), and d/dself masks out every
//! written position.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn scatter(&self, dim: usize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let op = "scatter";
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op,
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        if self.dtype() != DType::F32 || src.dtype() != DType::F32 {
            return Err(Error::DtypeMismatch {
                op,
                expected: DType::F32,
                got: if self.dtype() != DType::F32 {
                    self.dtype()
                } else {
                    src.dtype()
                },
            });
        }
        if self.device() != index.device() || self.device() != src.device() {
            return Err(Error::DeviceMismatch {
                op,
                lhs: self.device(),
                rhs: if self.device() != index.device() {
                    index.device()
                } else {
                    src.device()
                },
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
        // Flat target for every (k, src[k]) pair; duplicates allowed.
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
        let mut y = x.clone();
        // winner[t] = last k that wrote flat position t (usize::MAX = none).
        let mut winner = vec![usize::MAX; y.len()];
        for (k, (&t, &sv)) in targets.iter().zip(&s).enumerate() {
            y[t] = sv;
            winner[t] = k;
        }
        let out = Tensor::from_vec(y, &in_shape)?;
        if !self.requires_grad() && !src.requires_grad() {
            return Ok(out);
        }

        Ok(
            out.record_fn(vec![self.clone(), index.clone(), src.clone()], move |g| {
                let gd = g.to_vec();
                let mut gx = gd.clone();
                let mut gs = vec![0.0f32; s.len()];
                for (t, &w) in winner.iter().enumerate() {
                    if w == usize::MAX {
                        continue;
                    }
                    gx[t] = 0.0;
                    gs[w] = gd[t];
                }
                vec![
                    Tensor::from_vec(gx, &in_shape).unwrap(),
                    Tensor::zeros(&idx_shape),
                    Tensor::from_vec(gs, &idx_shape).unwrap(),
                ]
            }),
        )
    }
}
