//! `triu`: upper-triangular mask on a rank-2 matrix. Elements with
//! col - row >= diagonal are kept, the rest zeroed (torch semantics).
//! Backward passes the gradient through the same mask.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn triu(&self, diagonal: i64) -> Result<Tensor> {
        masked_tri(self, diagonal, true)
    }

    pub fn tril(&self, diagonal: i64) -> Result<Tensor> {
        masked_tri(self, diagonal, false)
    }
}

fn masked_tri(t: &Tensor, diagonal: i64, upper: bool) -> Result<Tensor> {
    let op = if upper { "triu" } else { "tril" };
    let shape = t.shape().to_vec();
    if shape.len() != 2 {
        return Err(Error::Unsupported {
            op,
            msg: format!("expected rank-2 matrix, got rank {}", shape.len()),
        });
    }
    let c = shape[1];
    let x = t.to_vec();
    let keep = move |i: usize, j: usize| {
        if upper {
            j as i64 - i as i64 >= diagonal
        } else {
            j as i64 - i as i64 <= diagonal
        }
    };
    let r = shape[0];
    let y: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(k, &v)| if keep(k / c, k % c) { v } else { 0.0 })
        .collect();
    let out = Tensor::from_vec(y, &shape)?;
    if !t.requires_grad() {
        return Ok(out);
    }
    Ok(out.record_fn(vec![t.clone()], move |g| {
        let gd = g.to_vec();
        let dx: Vec<f32> = gd
            .iter()
            .enumerate()
            .map(|(k, &v)| if keep(k / c, k % c) { v } else { 0.0 })
            .collect();
        vec![Tensor::from_vec(dx, &shape).unwrap()]
    }))
}
