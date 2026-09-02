//! `fmod` operator (C-style, sign of dividend): y = a - trunc(a/b) * b, which
//! is exactly Rust's f32 `%`. Backward: dy/da = 1, dy/db = -trunc(a/b).

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn fmod(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("fmod", self, other, |a, b| a % b)?;
        let (x, y) = (self.detach_copy(), other.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let db = raw_binary("fmod_dydb", &x, &y, |a, b| -(a / b).trunc()).unwrap();
            let gb = raw_binary("fmod_bwb", g, &db, |gg, p| gg * p).unwrap();
            vec![unbroadcast(g, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
