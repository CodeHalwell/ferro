//! Typed exponential-linear activation with input-dependent dense adjoints.
use crate::{DType, Error, Result, Tensor};
use crate::ops_ext::add_alpha::Alpha;

pub(crate) fn scalar(alpha: Alpha) -> (f64, f32) {
    match alpha {
        Alpha::Bool(value) => (u8::from(value) as f64, u8::from(value) as f32),
        Alpha::I64(value) => (value as f64, value as f32),
        Alpha::U64(value) => (value as f64, value as f32),
        Alpha::F64(value) => (value, value as f32),
    }
}

pub(crate) fn apply(input: &Tensor, op: &'static str, alpha: Alpha, scale: f64, input_scale: f64, backward_scale: f64) -> Result<Tensor> {
    if crate::capture::is_recording() {
        return Err(Error::Unsupported { op, msg: "inference capture is unsupported".into() });
    }
    if !matches!(input.dtype(), DType::F32 | DType::F64) {
        return Err(Error::DtypeMismatch { op, expected: DType::F32, got: input.dtype() });
    }
    let (alpha64, alpha32) = scalar(alpha);
    if input.dtype() == DType::F32 && [alpha64, scale, input_scale].iter().any(|x| x.is_finite() && x.abs() > f32::MAX as f64) {
        return Err(Error::InvalidShape { op, msg: "activation coefficients cannot be converted to float32 without overflow".into() });
    }
    let (scale32, input32, backward32) = (scale as f32, input_scale as f32, backward_scale as f32);
    let (negative32, negative64) = (alpha32 * scale32, alpha64 * scale);
    let output = crate::typed_math::pointwise_real(input, op,
        |x| if x < 0.0 { (x * input32).exp_m1() * negative32 } else { x * scale32 },
        |x| if x < 0.0 { (x * input_scale).exp_m1() * negative64 } else { x * scale })?;
    let saved = input.detach_copy();
    let backward = move |gradient: &Tensor| {
        let result = if saved.dtype() == DType::F64 {
            let values = gradient.to_vec_f64().into_iter().zip(saved.to_vec_f64()).map(|(g, x)| {
                if x <= 0.0 { ((g * backward_scale) * negative64) * (x * backward_scale).exp() } else { g * scale }
            }).collect();
            Tensor::from_vec_f64(values, saved.shape()).unwrap()
        } else {
            crate::tensor::raw_binary(op, gradient, &saved, |g, x| {
                if x <= 0.0 { ((g * backward32) * negative32) * (x * backward32).exp() } else { g * scale32 }
            }).unwrap()
        };
        vec![result]
    };
    Ok(if op == "celu" {
        output.record_fn_forward(vec![input.clone()], crate::autograd::ForwardOp::Celu(alpha), backward)
    } else { output.record_fn(vec![input.clone()], backward) })
}

impl Tensor {
    pub fn elu(&self, alpha: f32) -> Result<Tensor> {
        self.elu_alpha(Alpha::F64(alpha as f64))
    }

    pub fn elu_alpha(&self, alpha: Alpha) -> Result<Tensor> {
        apply(self, "elu", alpha, 1.0, 1.0, 1.0)
    }
}
