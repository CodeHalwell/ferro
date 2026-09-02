//! `lerp` with a scalar weight: y = a + w*(b-a), matching torch.lerp(input,
//! end, weight) for scalar weight. Backward: dy/da = 1-w, dy/db = w, so
//! da = g*(1-w), db = g*w.

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn lerp(&self, end: &Tensor, weight: f32) -> Result<Tensor> {
        let out = raw_binary("lerp", self, end, move |a, b| a + weight * (b - a))?;
        let (sx, sy) = (self.shape().to_vec(), end.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), end.clone()], move |g| {
            let ga = raw_binary("lerp_bwa", g, g, move |gg, _| gg * (1.0 - weight)).unwrap();
            let gb = raw_binary("lerp_bwb", g, g, move |gg, _| gg * weight).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
