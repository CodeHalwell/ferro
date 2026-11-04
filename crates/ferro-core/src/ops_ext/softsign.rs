//! Softsign with typed, staged quotient adjoints.

use crate::{DType, Error, Result, Tensor};

impl Tensor {
    pub fn softsign(&self) -> Result<Tensor> {
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op: "softsign", msg: "inference capture is not supported".into() });
        }
        let output = crate::typed_math::pointwise_real(self, "softsign", |x| x / (1.0 + x.abs()), |x| x / (1.0 + x.abs()))?
            .to_device(self.device())?;
        let saved = self.detach_copy();
        Ok(output.record_fn(vec![self.clone()], move |gradient| {
            let result = if saved.dtype() == DType::F64 {
                let values = gradient.to_vec_f64().into_iter().zip(saved.to_vec_f64()).map(|(g, x)| {
                    let d = 1.0 + x.abs();
                    let sign = if x < 0.0 { -1.0 } else if x > 0.0 { 1.0 } else { 0.0 };
                    g / d + ((-g) * ((x / d) / d)) * sign
                }).collect();
                Tensor::from_vec_f64(values, saved.shape()).unwrap()
            } else {
                crate::tensor::raw_binary("softsign_backward", gradient, &saved, |g, x| {
                    let d = 1.0 + x.abs();
                    let sign = if x < 0.0 { -1.0 } else if x > 0.0 { 1.0 } else { 0.0 };
                    g / d + ((-g) * ((x / d) / d)) * sign
                }).unwrap()
            };
            vec![result]
        }))
    }
}
