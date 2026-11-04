//! Sign, with zero at zero and NaN. Its recorded derivative is explicitly zero.
use crate::dispatch::{OpTag, UnaryKind};
use crate::error::Result;
use crate::tensor::{raw_unary_k, Tensor};

impl Tensor {
    pub fn sign(&self) -> Result<Tensor> {
        let out = raw_unary_k(self, UnaryKind::Sign)?;
        let shape = self.shape().to_vec();
        let device = self.device();
        let dtype = self.dtype();
        Ok(out.record_fn_tagged(vec![self.clone()], OpTag::Unary(UnaryKind::Sign), move |_| {
            vec![if dtype == crate::DType::F64 { crate::typed_math::full_f64(&shape, 0.0).unwrap() }
                 else { Tensor::full_on(&shape, 0.0, device).unwrap() }]
        }))
    }
}
