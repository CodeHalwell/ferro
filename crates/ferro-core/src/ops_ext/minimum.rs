//! Elementwise minimum: y = min(a, b), broadcasting over both operands
//! (torch.minimum semantics). Backward: gradient flows to whichever operand
//! is smaller (partial 1 for it, 0 for the other); at a tie (a == b) it
//! splits 0.5/0.5.

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("minimum", self, other, |a, b| a.min(b))?;
        let (x, y) = (self.detach_copy(), other.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let pa = raw_binary("minimum_dx", &x, &y, |a, b| {
                if a < b {
                    1.0
                } else if a > b {
                    0.0
                } else {
                    0.5
                }
            })
            .unwrap();
            let pb = raw_binary("minimum_dy", &x, &y, |a, b| {
                if b < a {
                    1.0
                } else if b > a {
                    0.0
                } else {
                    0.5
                }
            })
            .unwrap();
            let ga = raw_binary("minimum_bwa", g, &pa, |gg, p| gg * p).unwrap();
            let gb = raw_binary("minimum_bwb", g, &pb, |gg, p| gg * p).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
