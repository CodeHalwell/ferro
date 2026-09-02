//! sinh: y = sinh(x) = (e^x - e^-x)/2. Backward: d/dx sinh(x) = cosh(x),
//! so dx = g * cosh(x).

use crate::error::Result;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn sinh(&self) -> Result<Tensor> {
        let out = raw_binary("sinh", self, self, |v, _| v.sinh())?;
        let x = self.detach_copy();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("sinh_bw", g, &x, |gg, xx| gg * xx.cosh()).unwrap()]
        }))
    }
}
