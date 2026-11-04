//! SiLU (swish): y = x * sigmoid(x). Backward: with s = sigmoid(x),
//! dy/dx = s * (1 + x * (1 - s)).
//!
//! The forward routes through `UnaryKind::Silu` (device unary_dev kernel for
//! resident tensors); the backward is composed from tensor ops on g's device.

use crate::dispatch::UnaryKind;
use crate::tensor::{raw_unary_k, Tensor};

impl Tensor {
    pub fn silu(&self) -> Tensor {
        if self.dtype() == crate::DType::F64 {
            return crate::typed_math::unary_grad(self, "silu", |x| x / (1.0 + (-x).exp()), |g, x| {
                let s = 1.0 / (1.0 + (-x).exp());
                g * s * (1.0 + x * (1.0 - s))
            }).expect("float64 silu requires eager CPU tensors");
        }
        let out = raw_unary_k(self, UnaryKind::Silu)
            .expect("tensor's device backend is always registered");
        if !self.requires_grad() && !crate::forward_ad::is_active() {
            return if crate::capture::is_recording() {
                out.record_fn_tagged(vec![self.clone()], crate::OpTag::Unary(UnaryKind::Silu), |_| unreachable!("inference node has no backward"))
            } else { out };
        }
        let x = self.detach_copy();
        out.record_fn_tagged(
            vec![self.clone()],
            crate::OpTag::Unary(UnaryKind::Silu),
            move |g| {
            let dev = g.device();
            let c = |v: f32| Tensor::scalar(v).to_device(dev).unwrap();
            let one = c(1.0);
            let s = x.sigmoid();
            // dx = g * s * (1 + x * (1 - s))
            let inner = x.mul(&one.sub(&s).unwrap()).unwrap().add(&one).unwrap();
            vec![g.mul(&s).unwrap().mul(&inner).unwrap()]
        })
    }
}
