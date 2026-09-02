//! `maximum`: y = max(a, b), elementwise with broadcasting (torch.maximum).
//! Backward: gradient flows to whichever operand is larger; at a tie (a == b)
//! it splits 0.5/0.5, matching torch. da = g * (a>=b ? (a>b?1:0.5) : 0),
//! db = g * (b>=a ? (b>a?1:0.5) : 0), each unbroadcast to its input's shape.

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("maximum", self, other, |a, b| a.max(b))?;
        let (x, y) = (self.detach_copy(), other.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let pa = raw_binary("maximum_pa", &x, &y, |a, b| {
                if a > b {
                    1.0
                } else if a < b {
                    0.0
                } else {
                    0.5
                }
            })
            .unwrap();
            let pb = raw_binary("maximum_pb", &x, &y, |a, b| {
                if b > a {
                    1.0
                } else if b < a {
                    0.0
                } else {
                    0.5
                }
            })
            .unwrap();
            let ga = raw_binary("maximum_bwa", g, &pa, |gg, p| gg * p).unwrap();
            let gb = raw_binary("maximum_bwb", g, &pb, |gg, p| gg * p).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
