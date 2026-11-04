//! Round to nearest even, with optional decimal scaling in the input dtype.
use crate::{Result, Tensor};

impl Tensor {
    pub fn round(&self) -> Result<Tensor> {
        crate::typed_math::rounding(self, crate::typed_math::RoundingKind::Round(None))
    }

    pub fn round_decimals(&self, decimals: i64) -> Result<Tensor> {
        crate::typed_math::rounding(self, crate::typed_math::RoundingKind::Round(Some(decimals)))
    }
}
