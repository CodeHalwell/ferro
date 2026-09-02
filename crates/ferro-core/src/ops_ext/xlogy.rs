//! `xlogy`: y = a * ln(b), with torch's convention that y = 0 wherever a == 0
//! (even if b == 0), and y = NaN wherever b is NaN. Backward: da = g * ln(b),
//! db = g * a / b.

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn xlogy(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("xlogy", self, other, |a, b| {
            if b.is_nan() {
                f32::NAN
            } else if a == 0.0 {
                0.0
            } else {
                a * b.ln()
            }
        })?;
        let (x, y) = (self.detach_copy(), other.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let dlnb = raw_binary("xlogy_da", &x, &y, |_, b| b.ln()).unwrap();
            let ga = raw_binary("xlogy_bwa", g, &dlnb, |gg, p| gg * p).unwrap();
            let ratio = raw_binary("xlogy_db", &x, &y, |a, b| a / b).unwrap();
            let gb = raw_binary("xlogy_bwb", g, &ratio, |gg, p| gg * p).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
