//! Thresholded softplus with input-dtype beta arithmetic and dense adjoints.

use crate::{DType, Error, Result, Tensor};

impl Tensor {
    pub fn softplus(&self) -> Result<Tensor> {
        self.softplus_options(1.0, 20.0)
    }

    pub fn softplus_options(&self, beta: f64, threshold: f64) -> Result<Tensor> {
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op: "softplus", msg: "inference capture is not supported".into() });
        }
        if self.dtype() == DType::F32 && [beta, threshold].iter().any(|value| value.is_finite() && value.abs() > f32::MAX as f64) {
            return Err(Error::InvalidShape { op: "softplus", msg: "beta and threshold must be representable in float32".into() });
        }
        let beta32 = beta as f32;
        let threshold32 = threshold as f32;
        let output = crate::typed_math::pointwise_real(self, "softplus", |x| {
            let scaled = x * beta32;
            if scaled > threshold32 { x } else { scaled.exp().ln_1p() / beta32 }
        }, |x| {
            let scaled = x * beta;
            if scaled > threshold { x } else { scaled.exp().ln_1p() / beta }
        })?;
        let saved = self.detach_copy();
        Ok(output.record_fn(vec![self.clone()], move |gradient| {
            let result = if saved.dtype() == DType::F64 {
                let values = gradient.to_vec_f64().into_iter().zip(saved.to_vec_f64()).map(|(g, x)| {
                    let scaled = x * beta;
                    if scaled > threshold { g } else { let z = scaled.exp(); (g * z) / (z + 1.0) }
                }).collect();
                Tensor::from_vec_f64(values, saved.shape()).unwrap()
            } else {
                crate::tensor::raw_binary("softplus_backward", gradient, &saved, |g, x| {
                    let scaled = x * beta32;
                    if scaled > threshold32 { g } else { let z = scaled.exp(); (g * z) / (z + 1.0) }
                }).unwrap()
            };
            vec![result]
        }))
    }
}
