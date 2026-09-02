//! round: y = round_ties_even(x) (banker's rounding, matches torch/IEEE).
//! Backward: piecewise-constant almost everywhere, so dx = 0.

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn round(&self) -> Result<Tensor> {
        let out = raw_binary("round", self, self, |v, _| v.round_ties_even())?.to_device(self.device())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("round_bw", g, &x, |_, _| 0.0).unwrap()]
        }))
    }
}
