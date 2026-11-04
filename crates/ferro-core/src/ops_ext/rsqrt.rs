//! Reciprocal square root. Its adjoint reads the saved output, not the input.
use crate::{DType, Device, Error, Result, Tensor};

impl Tensor {
    pub fn rsqrt(&self) -> Result<Tensor> {
        if matches!(self.dtype(), DType::Bool | DType::I64) {
            if self.device() != Device::Cpu {
                return Err(Error::Unsupported { op: "rsqrt", msg: "integer reciprocal square root requires CPU storage".into() });
            }
            return self.to_dtype_autograd(DType::F32, false)?.rsqrt();
        }
        let out = crate::typed_math::pointwise_real(self, "rsqrt", |value| 1.0 / value.sqrt(), |value| 1.0 / value.sqrt())?;
        let saved = out.detach_copy();
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Rsqrt, move |gradient| {
            let derivative = if saved.dtype() == DType::F64 {
                Tensor::from_vec_f64(gradient.to_vec_f64().into_iter().zip(saved.to_vec_f64())
                    .map(|(g, y)| (g * -0.5) * y.powi(3)).collect(), saved.shape()).unwrap()
            } else {
                crate::tensor::raw_binary("rsqrt_backward", gradient, &saved, |g, y| (g * -0.5) * y.powi(3)).unwrap()
            };
            vec![derivative]
        }).save_output_version())
    }
}
