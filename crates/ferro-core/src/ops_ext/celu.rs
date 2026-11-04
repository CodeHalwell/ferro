//! CELU preserves separate forward and backward scalar conversion stages.
use crate::{Error, Result, Tensor};
use crate::ops_ext::add_alpha::Alpha;

impl Tensor {
    pub fn celu(&self, alpha: f32) -> Result<Tensor> {
        self.celu_alpha(Alpha::F64(alpha as f64))
    }

    pub fn celu_alpha(&self, alpha: Alpha) -> Result<Tensor> {
        let (value, value32) = super::elu::scalar(alpha);
        if value == 0.0 {
            return Err(Error::InvalidShape { op: "celu", msg: "alpha must be nonzero".into() });
        }
        super::elu::apply(self, "celu", alpha, 1.0, 1.0 / value, 1.0 / value32 as f64)
    }
}

pub(crate) fn validate_backward(alpha: Alpha) -> Result<()> {
    let (value, _) = super::elu::scalar(alpha);
    if value.is_finite() && value.abs() > f32::MAX as f64 {
        return Err(Error::InvalidShape { op: "celu backward", msg: "alpha cannot be converted to float32 without overflow".into() });
    }
    Ok(())
}
