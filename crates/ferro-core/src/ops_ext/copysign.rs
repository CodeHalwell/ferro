//! `copysign`: y = |a| with the sign of b, via f32::copysign. Backward:
//! dy/da = 1 when a and b share a sign, else -1 (undefined at a==0 or
//! b==0, matching torch); dy/db = 0.

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn copysign(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("copysign", self, other, |a, b| a.copysign(b))?.to_device(self.device())?;
        let (x, y) = (self.detach_copy(), other.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let da = raw_binary("copysign_da", &x, &y, |a, b| {
                if (a >= 0.0) == (b >= 0.0) {
                    1.0
                } else {
                    -1.0
                }
            })
            .unwrap();
            let ga = raw_binary("copysign_bwa", g, &da, |gg, p| gg * p).unwrap();
            let gb = raw_binary("copysign_bwb", g, &y, |_, _| 0.0).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
