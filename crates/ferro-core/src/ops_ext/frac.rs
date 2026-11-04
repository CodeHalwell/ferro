//! Fractional part, with Torch's identity adjoint even at integral inputs.
use crate::{Result, Tensor};

impl Tensor {
    pub fn frac(&self) -> Result<Tensor> {
        let out = crate::typed_math::pointwise_real(self, "frac", |value| value - value.trunc(), |value| value - value.trunc())?;
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Frac, |gradient| vec![gradient.clone()]))
    }
}
