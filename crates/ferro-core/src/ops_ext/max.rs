//! Global `max` reduction to a scalar. Forward returns the maximum over all
//! elements with torch semantics: empty input is an error and NaN propagates
//! (the first NaN encountered wins). Backward matches torch's
//! evenly_distribute_backward: the incoming scalar grad is split evenly across
//! every element equal to the max, so ties share it; a NaN result switches
//! the mask to isnan(input) so the gradient routes to the NaN entries, as in
//! torch's evenly_distribute_backward.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn max(&self) -> Result<Tensor> {
        if self.dtype() == crate::DType::Bool {
            if self.numel() == 0 { return Err(Error::InvalidShape { op: "max", msg: "cannot reduce an empty tensor".into() }); }
            return self.bool_reduce(None, false, false);
        }
        if self.dtype() == crate::DType::I64 {
            crate::typed_math::check("max", &[self], crate::DType::I64)?;
            let value = self.to_vec_i64().into_iter().max().ok_or_else(|| Error::InvalidShape { op: "max", msg: "cannot reduce an empty tensor".into() })?;
            return Tensor::from_vec_i64(vec![value], &[]);
        }
        if self.dtype() == crate::DType::F64 {
            crate::typed_math::check("max", &[self], crate::DType::F64)?;
            let values = self.to_vec_f64();
            let mut value = *values.first().ok_or_else(|| Error::InvalidShape { op: "max", msg: "cannot reduce an empty tensor".into() })?;
            for &v in &values { if v.is_nan() { value = v; break; } if v > value { value = v; } }
            let ties: Vec<_> = values.iter().enumerate().filter_map(|(i, &v)| (v == value || (value.is_nan() && v.is_nan())).then_some(i)).collect();
            let shape = self.shape().to_vec();
            return Ok(Tensor::from_vec_f64(vec![value], &[])?.record_fn(vec![self.clone()], move |g| {
                let mut out = crate::typed_math::filled("max backward", &shape, 0.0f64).unwrap();
                let share = g.to_vec_f64()[0] / ties.len() as f64;
                for &i in &ties { out[i] = share; }
                vec![Tensor::from_vec_f64(out, &shape).unwrap()]
            }).save_output_version());
        }
        crate::tensor::check_f32("max", self)?;
        let xv = self.to_vec();
        if xv.is_empty() {
            return Err(Error::InvalidShape {
                op: "max",
                msg: "cannot reduce an empty tensor".into(),
            });
        }
        let mut m = xv[0];
        for &v in &xv {
            if v.is_nan() {
                m = v;
                break;
            }
            if v > m {
                m = v;
            }
        }
        // Capture only the tie indices, not the materialized input: the closure
        // lives as long as the output, and xv would pin a full copy in memory.
        // A NaN max matches torch's evenly_distribute_backward: the mask
        // becomes isnan(input), so the gradient routes to the NaN entries.
        let tie = |v: f32| if m.is_nan() { v.is_nan() } else { v == m };
        let ties: Vec<usize> = xv
            .iter()
            .enumerate()
            .filter(|(_, &v)| tie(v))
            .map(|(i, _)| i)
            .collect();
        let len = xv.len();
        let out = Tensor::scalar(m);
        let shape = self.shape().to_vec();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let mut grad = vec![0.0; len];
            if !ties.is_empty() {
                let share = g.item() / ties.len() as f32;
                for &i in &ties {
                    grad[i] = share;
                }
            }
            vec![Tensor::from_vec(grad, &shape).unwrap()]
        }).save_output_version())
    }
}
