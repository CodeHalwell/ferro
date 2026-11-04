//! SELU uses the original double constants before input-dtype conversion.
use crate::{Result, Tensor};
use crate::ops_ext::add_alpha::Alpha;

impl Tensor {
    pub fn selu(&self) -> Result<Tensor> {
        super::elu::apply(self, "selu", Alpha::F64(1.6732632423543772848170429916717),
            1.0507009873554804934193349852946, 1.0, 1.0)
    }
}
