//! Stable log-sigmoid with input-dtype first-order adjoints.

use crate::{DType, Error, Result, Tensor};

impl Tensor {
    pub fn log_sigmoid(&self) -> Result<Tensor> {
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op: "log_sigmoid", msg: "inference capture is not supported".into() });
        }
        let output = crate::typed_math::pointwise_real(self, "log_sigmoid", |x| {
            let lower = if x < 0.0 { x } else { 0.0 };
            lower - (-x.abs()).exp().ln_1p()
        }, |x| {
            let lower = if x < 0.0 { x } else { 0.0 };
            lower - (-x.abs()).exp().ln_1p()
        })?.contiguous()?;
        let saved = self.detach_copy();
        Ok(output.record_fn(vec![self.clone()], move |gradient| {
            let result = if saved.dtype() == DType::F64 {
                let values = gradient.to_vec_f64().into_iter().zip(saved.to_vec_f64()).map(|(g, x)| {
                    let buffer = (-x.abs()).exp();
                    let (maximum, sign) = if x < 0.0 { (1.0, 1.0) } else { (0.0, -1.0) };
                    (maximum - sign * (buffer / (1.0 + buffer))) * g
                }).collect();
                Tensor::from_vec_f64(values, saved.shape()).unwrap()
            } else {
                crate::tensor::raw_binary("log_sigmoid_backward", gradient, &saved, |g, x| {
                    let buffer = (-x.abs()).exp();
                    let (maximum, sign) = if x < 0.0 { (1.0, 1.0) } else { (0.0, -1.0) };
                    (maximum - sign * (buffer / (1.0 + buffer))) * g
                }).unwrap()
            };
            vec![result]
        }))
    }
}
