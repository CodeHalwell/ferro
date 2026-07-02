//! Global `max` reduction to a scalar. Forward returns the maximum over all
//! elements with torch semantics: empty input is an error and NaN propagates
//! (the first NaN encountered wins). Backward routes the incoming scalar grad
//! to the argmax position captured at forward time (ties break toward the
//! lowest flat index) and 0 elsewhere.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn max(&self) -> Result<Tensor> {
        let xv = self.to_vec();
        if xv.is_empty() {
            return Err(Error::InvalidShape { op: "max", msg: "cannot reduce an empty tensor".into() });
        }
        let mut arg = 0usize;
        for (i, &v) in xv.iter().enumerate() {
            if v.is_nan() {
                arg = i;
                break;
            }
            if v > xv[arg] {
                arg = i;
            }
        }
        let out = Tensor::scalar(xv[arg]);
        let shape = self.shape().to_vec();
        let n = xv.len();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let mut grad = vec![0.0; n];
            grad[arg] = g.item();
            vec![Tensor::from_vec(grad, &shape).unwrap()]
        }))
    }
}
