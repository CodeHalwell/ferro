//! `cat` concatenation along a dimension. Row-major layout: for each `outer`
//! index (prod of dims before `dim`), each input contributes a contiguous
//! block of `shape[dim] * inner` elements (inner = prod of dims after `dim`).

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn cat(tensors: &[Tensor], dim: usize) -> Result<Tensor> {
        let first = tensors.first().ok_or_else(|| Error::InvalidShape {
            op: "cat",
            msg: "expected a non-empty list of tensors".into(),
        })?;
        let base = first.shape().to_vec();
        if dim >= base.len() {
            return Err(Error::InvalidShape {
                op: "cat",
                msg: format!("dim {dim} out of range for rank {}", base.len()),
            });
        }
        for t in &tensors[1..] {
            let s = t.shape();
            let same_rank = s.len() == base.len();
            let ok = same_rank
                && s.iter().zip(&base).enumerate().all(|(d, (a, b))| d == dim || a == b);
            if !ok {
                return Err(Error::ShapeMismatch {
                    op: "cat",
                    lhs: base.clone(),
                    rhs: s.to_vec(),
                });
            }
        }

        let outer: usize = base[..dim].iter().product();
        let inner: usize = base[dim + 1..].iter().product();
        let dim_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[dim]).collect();
        let cat_size: usize = dim_sizes.iter().sum();

        let mut out_shape = base;
        out_shape[dim] = cat_size;
        let mut out_data = vec![0.0f32; outer * cat_size * inner];
        let mut offset = 0usize;
        for (t, &d) in tensors.iter().zip(&dim_sizes) {
            let src = t.to_vec();
            let block = d * inner;
            for o in 0..outer {
                let dst = o * cat_size * inner + offset * inner;
                out_data[dst..dst + block].copy_from_slice(&src[o * block..(o + 1) * block]);
            }
            offset += d;
        }
        let out = Tensor::from_vec(out_data, &out_shape)?;

        let in_shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape().to_vec()).collect();
        Ok(out.record_fn(tensors.to_vec(), move |g| {
            let g_data = g.to_vec();
            let mut grads = Vec::with_capacity(in_shapes.len());
            let mut offset = 0usize;
            for (shape, &d) in in_shapes.iter().zip(&dim_sizes) {
                let block = d * inner;
                let mut gi = vec![0.0f32; outer * block];
                for o in 0..outer {
                    let src = o * cat_size * inner + offset * inner;
                    gi[o * block..(o + 1) * block].copy_from_slice(&g_data[src..src + block]);
                }
                grads.push(Tensor::from_vec(gi, shape).unwrap());
                offset += d;
            }
            grads
        }))
    }
}
