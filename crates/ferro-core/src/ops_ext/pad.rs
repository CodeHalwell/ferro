//! Constant padding along every dimension. `pads` holds one (before, after)
//! pair per dimension in dim order: [before_0, after_0, before_1, after_1, ..].
//! Backward slices the output gradient back to the input extent.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn pad_constant(&self, pads: &[usize], value: f32) -> Result<Tensor> {
        let op = "pad_constant";
        let pads = pads.to_vec();
        let ndim = self.ndim();
        if pads.len() != 2 * ndim {
            return Err(Error::InvalidShape {
                op,
                msg: format!(
                    "expected {} pad values ({ndim} dims x before/after), got {}",
                    2 * ndim,
                    pads.len()
                ),
            });
        }
        let pads = pads.to_vec();
        let in_shape = self.shape().to_vec();
        let out_shape: Vec<usize> = (0..ndim)
            .map(|d| in_shape[d] + pads[2 * d] + pads[2 * d + 1])
            .collect();

        // Flat offset of each input element inside the padded buffer.
        let out_numel: usize = out_shape.iter().product();
        let mut y = vec![value; out_numel];
        let x = self.to_vec();
        let mut coord_in = vec![0usize; ndim];
        let numel = x.len();
        for &xi in x.iter().take(numel) {
            let mut off = 0usize;
            for d in 0..ndim {
                off = off * out_shape[d] + coord_in[d] + pads[2 * d];
            }
            y[off] = xi;
            for d in (0..ndim).rev() {
                coord_in[d] += 1;
                if coord_in[d] < in_shape[d] {
                    break;
                }
                coord_in[d] = 0;
            }
        }
        let out = Tensor::from_vec(y, &out_shape)?;
        if !self.requires_grad() {
            return Ok(out);
        }

        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let mut dx = vec![0.0f32; numel];
            let mut coord = vec![0usize; ndim];
            for i in 0..numel {
                let mut off = 0usize;
                for d in 0..ndim {
                    off = off * out_shape[d] + coord[d] + pads[2 * d];
                }
                dx[i] = gd[off];
                for d in (0..ndim).rev() {
                    coord[d] += 1;
                    if coord[d] < in_shape[d] {
                        break;
                    }
                    coord[d] = 0;
                }
            }
            vec![Tensor::from_vec(dx, &in_shape).unwrap()]
        }))
    }
}
