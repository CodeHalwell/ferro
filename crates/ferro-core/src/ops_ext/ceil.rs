//! Typed ceil; its derivative is zero, including at discontinuities.
use crate::{Result, Tensor};

impl Tensor {
    pub fn ceil(&self) -> Result<Tensor> {
        crate::typed_math::rounding(self, crate::typed_math::RoundingKind::Ceil)
    }
}
