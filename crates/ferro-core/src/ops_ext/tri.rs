//! `triu`: upper-triangular mask on the final two dimensions. Elements with
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
    if shape.len() < 2 {
        return Err(Error::Unsupported {
            op,
            msg: format!("expected at least two dimensions, got rank {}", shape.len()),
        });
    }
    let (rows, c) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    let keep = move |i: usize, j: usize| {
        if upper {
            j as i128 - i as i128 >= diagonal as i128
        } else {
            j as i128 - i as i128 <= diagonal as i128
        }
    };
    if matches!(t.dtype(), crate::DType::F64 | crate::DType::I64) {
        crate::typed_math::check(op, &[t], t.dtype())?;
        if t.dtype() == crate::DType::I64 {
            return Tensor::from_vec_i64(t.to_vec_i64().into_iter().enumerate()
                .map(|(k, v)| if keep((k / c) % rows, k % c) { v } else { 0 }).collect(), &shape);
        }
        let data = t.to_vec_f64().into_iter().enumerate()
            .map(|(k, v)| if keep((k / c) % rows, k % c) { v } else { 0.0 }).collect();
        return Ok(Tensor::from_vec_f64(data, &shape)?.record_fn(vec![t.clone()], move |g| {
            let data = g.to_vec_f64().into_iter().enumerate()
                .map(|(k, v)| if keep((k / c) % rows, k % c) { v } else { 0.0 }).collect();
            vec![Tensor::from_vec_f64(data, &shape).unwrap()]
        }));
    }
    if t.dtype() == crate::DType::Bool {
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op, msg: "boolean operations do not support inference capture".into() });
        }
        return Tensor::from_vec_bool(t.to_vec_bool().into_iter().enumerate()
            .map(|(k, v)| v && keep((k / c) % rows, k % c)).collect(), &shape);
    }
    crate::tensor::check_f32(op, t)?;
    let x = t.to_vec();
    let y: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(k, &v)| if keep((k / c) % rows, k % c) { v } else { 0.0 })
        .collect();
    let out = Tensor::from_vec(y, &shape)?;
    if !t.requires_grad() && !crate::capture::is_recording() && !crate::forward_ad::is_active() {
        return Ok(out);
    }
    Ok(out.record_fn(vec![t.clone()], move |g| {
        let gd = g.to_vec();
        let dx: Vec<f32> = gd
            .iter()
            .enumerate()
            .map(|(k, &v)| if keep((k / c) % rows, k % c) { v } else { 0.0 })
            .collect();
        vec![Tensor::from_vec(dx, &shape).unwrap()]
    }))
}
