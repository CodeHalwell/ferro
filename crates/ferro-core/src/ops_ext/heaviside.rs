//! `heaviside`: y = 0 for x < 0, values for x == 0, 1 for x > 0 (torch.heaviside,
//! self is x, `values` supplies the x == 0 case). Backward: the true
//! derivative is 0 almost everywhere for both inputs (torch semantics), so
//! dx = 0 and dvalues = 0 regardless of g, each unbroadcast to its input's
//! shape.

use crate::error::Result;
use crate::tensor::{raw_binary, unbroadcast, Tensor};

impl Tensor {
    pub fn heaviside(&self, values: &Tensor) -> Result<Tensor> {
        let out = raw_binary("heaviside", self, values, |x, v| {
            if x < 0.0 {
                0.0
            } else if x > 0.0 {
                1.0
            } else {
                v
            }
        })?;
        let (sx, sv) = (self.shape().to_vec(), values.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), values.clone()], move |g| {
            let z = raw_binary("heaviside_bw", g, g, |_, _| 0.0).unwrap();
            vec![unbroadcast(&z, &sx), unbroadcast(&z, &sv)]
        }))
    }
}
