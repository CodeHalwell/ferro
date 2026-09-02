//! `remainder`: y = a - floor(a/b) * b, Python/torch modulo taking the sign
//! of the divisor (differs from `fmod`, which takes the sign of the
//! dividend). Backward: dy/da = 1, dy/db = -floor(a/b).

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn remainder(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("remainder", self, other, |a, b| a - (a / b).floor() * b)?;
        let (x, y) = (self.detach_copy(), other.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let pb = raw_binary("remainder_pb", &x, &y, |a, b| -(a / b).floor()).unwrap();
            let gb = raw_binary("remainder_bwb", g, &pb, |gg, p| gg * p).unwrap();
            vec![unbroadcast(g, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
