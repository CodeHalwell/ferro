//! Square root. Inputs assumed positive. Backward: d/dx sqrt(x) = 1/(2*sqrt(x)),
//! so with y = sqrt(x), dx = g / (2*y).

use crate::dispatch::UnaryKind;
use crate::tensor::{raw_binary, raw_unary_k, Tensor};

impl Tensor {
    pub fn sqrt(&self) -> Tensor {
        let out = raw_unary_k(self, UnaryKind::Sqrt).expect("cpu backend is always registered");
        if !self.requires_grad() && !crate::forward_ad::is_active() {
            return if crate::capture::is_recording() {
                out.record_fn_tagged(vec![self.clone()], crate::OpTag::Unary(UnaryKind::Sqrt), |_| unreachable!("inference node has no backward"))
            } else { out };
        }
        let y = out.detach_copy_layout().expect("validated square-root output snapshot");
        out.record_fn_tagged(vec![self.clone()], crate::OpTag::Unary(UnaryKind::Sqrt), move |g| {
            vec![if y.device() == crate::Device::Cpu {
                let two = if y.dtype() == crate::DType::F64 { crate::typed_math::full_f64(&[], 2.0).unwrap() } else { Tensor::scalar(2.0) };
                g.div(&y.mul(&two).unwrap()).unwrap()
            } else { raw_binary("sqrt_bw", g, &y, |gg, yy| gg / (2.0 * yy)).unwrap() }]
        }).save_output_version()
    }
}
