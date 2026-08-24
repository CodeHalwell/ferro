//! `gather` along one dimension by an I64 index tensor (torch semantics):
//! out[i][j][k] = self[i][index[i][j][k]][k] for dim=1, and analogously for
//! other dims. `index` must have the same rank as `self` and be no larger in
//! any non-`dim` dimension; the output takes `index`'s shape. Backward
//! scatter-adds grad back to the gathered positions (duplicates accumulate).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn gather(&self, dim: usize, index: &Tensor) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "gather",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        if index.dtype() != DType::I64 {
            return Err(Error::DtypeMismatch {
                op: "gather",
                expected: DType::I64,
                got: index.dtype(),
            });
        }
        if index.ndim() != ndim {
            return Err(Error::InvalidShape {
                op: "gather",
                msg: format!("index rank {} must match input rank {ndim}", index.ndim()),
            });
        }
        let in_shape = self.shape().to_vec();
        let out_shape = index.shape().to_vec();
        for d in 0..ndim {
            if d != dim && out_shape[d] > in_shape[d] {
                return Err(Error::InvalidShape {
                    op: "gather",
                    msg: format!(
                        "index shape {out_shape:?} exceeds input shape {in_shape:?} at dim {d}"
                    ),
                });
            }
        }

        let idx = index.to_vec_i64();
        let dim_size = in_shape[dim];
        // Flat source offset for each output position: walk output coordinates,
        // substituting the gathered index along `dim`.
        let mut src = Vec::with_capacity(idx.len());
        let mut coord = vec![0usize; ndim];
        for &id in &idx {
            if id < 0 || id as usize >= dim_size {
                return Err(Error::InvalidShape {
                    op: "gather",
                    msg: format!("index {id} out of range for dim {dim} with size {dim_size}"),
                });
            }
            let mut off = 0usize;
            for d in 0..ndim {
                let c = if d == dim { id as usize } else { coord[d] };
                off = off * in_shape[d] + c;
            }
            src.push(off);
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if coord[d] < out_shape[d] {
                    break;
                }
                coord[d] = 0;
            }
        }

        let x = self.to_vec();
        let y: Vec<f32> = src.iter().map(|&o| x[o]).collect();
        let out = Tensor::from_vec(y, &out_shape)?;
        if !self.requires_grad() {
            return Ok(out);
        }
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let mut gx = vec![0.0f32; in_shape.iter().product()];
            for (i, &o) in src.iter().enumerate() {
                gx[o] += gd[i];
            }
            vec![Tensor::from_vec(gx, &in_shape).unwrap()]
        }))
    }
}
