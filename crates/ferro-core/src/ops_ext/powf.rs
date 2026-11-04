//! `powf` operator: tensor raised to a scalar power. Forward: x^p.
//! Backward: d/dx x^p = p * x^(p-1), so dx = g * (p * x^(p-1)).

use crate::dispatch::UnaryKind;
use crate::tensor::{raw_binary, raw_unary_k, Tensor};

impl Tensor {
    /// Power by an original scalar exponent. Exact f32 exponents retain the
    /// registered f32 kernels; other exponents use a portable scalar CPU
    /// evaluation without narrowing the exponent or changing its domain.
    /// Device tensors use an explicit host fallback returning a CPU tensor
    /// for that latter case.
    pub fn pow_scalar(&self, p: f64) -> crate::Result<Tensor> {
        if self.dtype() == crate::DType::F64 { return self.powf64(p); }
        crate::tensor::check_f32("pow", self)?;
        let out = if p == p as f32 as f64 {
            raw_unary_k(self, UnaryKind::Powf(p as f32))?
        } else if self.device() == crate::Device::Cpu {
            crate::typed_math::pointwise_binary_real("pow", self, self,
                |x, _| (x as f64).powf(p) as f32, |_, _| unreachable!())?
        } else {
            if crate::capture::is_recording() {
                return Err(crate::Error::Unsupported { op: "pow", msg: "non-f32 scalar power uses a host fallback and cannot be device-captured".into() });
            }
            crate::typed_math::check_shape("pow", self.shape())?;
            raw_binary("pow", self, self, |x, _| (x as f64).powf(p) as f32)?
        };
        let input = self.clone();
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::ScalarPower(p), move |g| {
            // The ordinary engine transfers this final contribution back to
            // its leaf. Keep any unsupported device power visibly on CPU.
            let exponent = p - 1.0;
            let host = input.device() != crate::Device::Cpu
                && (g.device() == crate::Device::Cpu || exponent != exponent as f32 as f64);
            let device = if host { crate::Device::Cpu } else { input.device() };
            let source = input.to_device(device).expect("scalar power input transfer");
            let gradient = g.to_device(device).expect("scalar power cotangent transfer");
            vec![scalar_power_backward(&gradient, &source, p).expect("validated scalar power backward")]
        }))
    }

    pub fn powf(&self, p: f32) -> Tensor {
        if self.dtype() == crate::DType::F64 { return self.powf64(p as f64).expect("float64 power"); }
        let out = raw_unary_k(self, UnaryKind::Powf(p)).expect("cpu backend is always registered");
        let x = self.clone();
        out.record_fn_tagged(vec![self.clone()], crate::OpTag::Unary(UnaryKind::Powf(p)), move |g| {
            if p == 0.0 { return vec![Tensor::full_on(g.shape(), 0.0, g.device()).unwrap()]; }
            if g.device() == crate::Device::Cpu {
                let slope = x.powf(p - 1.0).mul(&Tensor::scalar(p)).unwrap();
                return vec![g.mul(&slope).unwrap()];
            }
            vec![raw_binary("powf_bw", g, &x, move |gg, xx| {
                let exponent = p - 1.0;
                let value = if exponent == 0.5 { xx.sqrt() } else if exponent == -0.5 { xx.sqrt().recip() } else if exponent == -1.0 { xx.recip() } else if exponent == -2.0 { (xx * xx).recip() } else if exponent == 2.0 { xx * xx } else if exponent == 3.0 { (xx * xx) * xx } else { xx.powf(exponent) };
                gg * (p * value)
            }).unwrap()]
        })
    }
}

pub(crate) fn scalar_power_backward(g: &Tensor, input: &Tensor, power: f64) -> crate::Result<Tensor> {
    if g.device() != input.device() {
        return Err(crate::Error::DeviceMismatch { op: "pow backward", lhs: g.device(), rhs: input.device() });
    }
    if power == 0.0 { return Tensor::full_on(input.shape(), 0.0, input.device()); }
    let exponent = power - 1.0;
    if input.device() != crate::Device::Cpu && exponent != exponent as f32 as f64 {
        return Err(crate::Error::Unsupported { op: "pow backward", msg: "graph differentiation through an original-exponent host fallback across devices is unsupported".into() });
    }
    // The exponent subtraction precedes narrowing. The resulting slope is
    // evaluated in the primal dtype before a possibly wider cotangent.
    let slope = input.pow_scalar(power - 1.0)?.mul(&Tensor::full_on(&[], power as f32, input.device())?)?;
    g.mul(&slope)
}
