//! Typed CPU acos with ordinary reverse (-g) * rsqrt(1 - x*x).
//! The existing F32 host-backed device fallback remains available.
use crate::tensor::{raw_binary, Tensor};
use crate::{DType, Device, Error, Result};

impl Tensor {
    pub fn acos(&self) -> Result<Tensor> {
        if !matches!(self.dtype(), DType::F32 | DType::F64) {
            return Err(Error::DtypeMismatch { op: "acos", expected: DType::F32, got: self.dtype() });
        }
        let dtype = self.dtype();
        let out = if self.device() == Device::Cpu {
            crate::typed_math::pointwise_binary_real("acos", self, self, |x, _| x.acos(), |x, _| x.acos())?
        } else {
            raw_binary("acos", self, self, |v, _| v.acos())?.to_device(self.device())?
        };
        let x = self.detach();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let g = g.detach().to_dtype_autograd(dtype, false).expect("validated acos cotangent cast");
            let out = if x.device() == Device::Cpu {
                crate::typed_math::pointwise_binary_real("acos_backward", &g, &x,
                    |g, x| (-g) * (1.0 - x*x).sqrt().recip(),
                    |g, x| (-g) * (1.0 - x*x).sqrt().recip())
            } else {
                raw_binary("acos_backward", &g, &x, |g, x| (-g) * (1.0 - x*x).sqrt().recip())
            };
            vec![out.expect("validated acos backward")]
        }))
    }
}
