//! `logaddexp`: y = ln(exp(a) + exp(b)), computed stably as
//! m + ln(exp(a-m) + exp(b-m)) with m = max(a, b). Backward:
//! da = g * sigmoid(a-b), db = g * sigmoid(b-a) = g * (1 - sigmoid(a-b)).

use crate::tensor::{raw_binary, unbroadcast, Tensor};
use crate::Result;

impl Tensor {
    pub fn logaddexp(&self, other: &Tensor) -> Result<Tensor> {
        let out = raw_binary("logaddexp", self, other, |a, b| {
            let m = a.max(b);
            m + ((a - m).exp() + (b - m).exp()).ln()
        })?;
        let (x, y) = (self.detach_copy(), other.detach_copy());
        let (sx, sy) = (self.shape().to_vec(), other.shape().to_vec());
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let da = raw_binary("logaddexp_da", &x, &y, |a, b| 1.0 / (1.0 + (b - a).exp())).unwrap();
            let ga = raw_binary("logaddexp_bwa", g, &da, |gg, p| gg * p).unwrap();
            let gb = raw_binary("logaddexp_bwb", g, &da, |gg, p| gg * (1.0 - p)).unwrap();
            vec![unbroadcast(&ga, &sx), unbroadcast(&gb, &sy)]
        }))
    }
}
