//! Hyperbolic tangent. Backward: d/dx tanh(x) = 1 - tanh(x)^2, so with
//! y = tanh(x) the gradient is dx = g * (1 - y^2).

use crate::dispatch::UnaryKind;
use crate::tensor::{raw_unary_k, Tensor};

impl Tensor {
    pub fn tanh(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Tanh).expect("cpu backend is always registered");
        if !self.requires_grad() && !crate::forward_ad::is_active() {
            return if crate::capture::is_recording() {
                out.record_fn_tagged(vec![self.clone()], crate::OpTag::Unary(UnaryKind::Tanh), |_| unreachable!("inference node has no backward"))
            } else { out };
        }
        let y = out.detach_copy_layout().expect("validated tanh output snapshot");
        out.record_fn_tagged(vec![self.clone()], crate::OpTag::Unary(UnaryKind::Tanh), move |g| {
            vec![crate::ops_ext::activation_backward::backward(crate::ops_ext::activation_backward::Kind::Tanh, g, &y).unwrap()]
        }).save_output_version()
    }
}
