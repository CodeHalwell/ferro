//! Natural logarithm. Reference implementation for the ops_ext pattern:
//! compute the value with a raw kernel, then record a self-contained backward
//! closure via `record_fn`. Backward: d/dx log(x) = 1/x, so dx = g / x.

use crate::tensor::{raw_binary, raw_unary, Tensor};

impl Tensor {
    pub fn log(&self) -> Tensor {
        let out = raw_unary(self, |x| x.ln());
        let x = self.clone();
        out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("log_bw", g, &x, |gg, xx| gg / xx).unwrap()]
        })
    }
}
