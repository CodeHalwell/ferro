//! SiLU (swish): y = x * sigmoid(x). Backward: with s = sigmoid(x),
//! dy/dx = s * (1 + x * (1 - s)).
//!
//! The forward routes through `UnaryKind::Silu` (device unary_dev kernel for
//! resident tensors); the backward is composed from tensor ops on g's device.

use crate::dispatch::UnaryKind;
use crate::tensor::{raw_unary_k, Tensor};

impl Tensor {
    pub fn silu(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Silu).expect("tensor's device backend is always registered");
        if !self.requires_grad() {
            return out;
        }
        let x = self.detach_copy();
        out.record_fn(vec![self.clone()], move |g| {
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
