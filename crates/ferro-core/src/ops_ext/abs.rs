//! `abs` operator. Backward: dx = g * sign(x), with sign(0) = 0
//! (avoid f32::signum, which returns +1 at 0).

use crate::dispatch::UnaryKind;
use crate::tensor::{raw_binary, raw_unary_k, Tensor};

impl Tensor {
    pub fn abs(&self) -> Tensor {
        if self.dtype() == crate::DType::F64 { return crate::typed_math::unary_grad_tensor(self, "abs", |x| x.abs(), false, |g, x| g.mul(&x.sign()?)).expect("float64 unary"); }
        let out = raw_unary_k(self, UnaryKind::Abs).expect("cpu backend is always registered");
        let x = self.clone();
        out.record_fn_tagged(vec![self.clone()], crate::OpTag::Unary(UnaryKind::Abs), move |g| {
            vec![raw_binary("abs_bw", g, &x, |gg, xx| {
                gg * if xx > 0.0 { 1.0 } else if xx < 0.0 { -1.0 } else { 0.0 }
            })
            .unwrap()]
        })
    }
}
