//! `hypot`: y = sqrt(a^2 + b^2), computed via f32::hypot for overflow safety.
//! Backward: dy/da = a/y, dy/db = b/y, so da = g*a/y, db = g*b/y.

use crate::tensor::{raw_binary, unbroadcast, Tensor};
use crate::Result;

impl Tensor {
    pub fn hypot(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("hypot", self, other, |a, b| a.hypot(b))?.to_device(self.device())?;
        let (x, y) = (self.detach_copy(), other.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        let yo = out.detach_copy();
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let da = raw_binary("hypot_da", &x, &yo, |a, yy| a / yy).unwrap();
            let db = raw_binary("hypot_db", &y, &yo, |b, yy| b / yy).unwrap();
            let ga = raw_binary("hypot_bwa", g, &da, |gg, p| gg * p).unwrap();
            let gb = raw_binary("hypot_bwb", g, &db, |gg, p| gg * p).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
