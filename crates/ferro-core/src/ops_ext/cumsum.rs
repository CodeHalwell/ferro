//! Typed cumulative sums and their linear reverse-prefix adjoints.
use crate::{DType, Device, Error, Result, Tensor};
use crate::autograd::ForwardOp;

#[derive(Clone, Copy)]
pub(crate) struct Geometry { pub outer: usize, pub width: usize, pub inner: usize, pub count: usize }

pub(crate) fn prepare(input: &Tensor, dim: usize, dtype: Option<DType>, op: &'static str, legacy: bool) -> Result<(Geometry, DType)> {
    if dim >= input.ndim().max(1) {
        return Err(Error::InvalidShape { op, msg: format!("dim {dim} out of range for rank {}", input.ndim()) });
    }
    if !matches!(input.dtype(), DType::Bool | DType::I64 | DType::F32 | DType::F64) {
        return Err(Error::DtypeMismatch { op, expected: DType::F32, got: input.dtype() });
    }
    let output = dtype.unwrap_or(if input.dtype() == DType::Bool { DType::I64 } else { input.dtype() });
    if !matches!(output, DType::I64 | DType::F32 | DType::F64) {
        return Err(Error::Unsupported { op, msg: "cumulative operations require int64, float32 or float64 output".into() });
    }
    let old_f32 = legacy && dtype.is_none() && input.dtype() == DType::F32;
    if crate::capture::is_recording() && !old_f32 {
        return Err(Error::Unsupported { op, msg: "typed cumulative operations do not support inference capture".into() });
    }
    if input.device() != Device::Cpu && !old_f32 {
        return Err(Error::Unsupported { op, msg: "typed cumulative operations require CPU inputs".into() });
    }
    if output == DType::I64 && matches!(input.dtype(), DType::F32 | DType::F64)
        && ((crate::autograd::is_grad_enabled() && crate::forward_ad::records_grad(input)) || crate::forward_ad::needs(&[input.clone()])) {
        return Err(Error::Unsupported { op, msg: "differentiable cumulative operations cannot produce int64".into() });
    }
    let count = crate::typed_math::check_shape(op, input.shape())?;
    if input.shape().iter().map(|size| (*size).max(1)).product::<usize>() > isize::MAX as usize {
        return Err(Error::InvalidShape { op, msg: "cumulative layout exceeds signed addressable capacity".into() });
    }
    let geometry = if input.ndim() == 0 { Geometry { outer: 1, width: 1, inner: 1, count } }
        else { Geometry { outer: input.shape()[..dim].iter().product(), width: input.shape()[dim], inner: input.shape()[dim + 1..].iter().product(), count } };
    Ok((geometry, output))
}

pub(crate) fn rounded(value: f64, dtype: DType) -> f64 { if dtype == DType::F32 { value as f32 as f64 } else { value } }

pub(crate) fn real_values(input: &Tensor, dtype: DType, op: &'static str) -> Result<Vec<f64>> {
    let converted = input.to_dtype(dtype);
    if dtype == DType::F64 { return Ok(converted.to_vec_f64()); }
    let mut values = crate::typed_math::filled(op, input.shape(), 0.0f64)?;
    for (output, value) in values.iter_mut().zip(converted.to_vec()) { *output = value as f64; }
    Ok(values)
}

pub(crate) fn real_tensor(values: Vec<f64>, shape: &[usize], dtype: DType, op: &'static str) -> Result<Tensor> {
    if dtype == DType::F64 { Tensor::from_vec_f64(values, shape) }
    else {
        let mut output = crate::typed_math::filled(op, shape, 0.0f32)?;
        for (out, value) in output.iter_mut().zip(values) { *out = value as f32; }
        Tensor::from_vec(output, shape)
    }
}

pub(crate) fn selected_vjp(g: &Tensor, shape: &[usize], dtype: DType, dim: usize, reverse: bool) -> Result<Tensor> {
    let width = shape.get(dim).copied().unwrap_or(1);
    let seed = g.to_dtype_autograd(dtype, false)?;
    if seed.numel() == 0 || width == 1 { Ok(seed) }
    else { seed.cumulative_sum(dim, None, !reverse) }
}

impl Tensor {
    pub fn cumsum(&self, dim: usize) -> Result<Tensor> { self.cumulative_sum(dim, None, false) }

    pub fn cumsum_dtype(&self, dim: usize, dtype: Option<DType>) -> Result<Tensor> { self.cumulative_sum(dim, dtype, false) }

    pub(crate) fn cumulative_sum(&self, dim: usize, dtype: Option<DType>, reverse: bool) -> Result<Tensor> {
        let (geometry, output_dtype) = prepare(self, dim, dtype, "cumsum", !reverse)?;
        let Geometry { outer, width, inner, count } = geometry;
        let output = if output_dtype == DType::I64 {
            let values = self.to_dtype(DType::I64).to_vec_i64();
            let mut output = crate::typed_math::filled("cumsum", self.shape(), 0i64)?;
            if count != 0 {
                for group in 0..outer { for lane in 0..inner {
                    let mut total = 0i64;
                    for step in 0..width {
                        let position = if reverse { width - 1 - step } else { step };
                        let index = (group * width + position) * inner + lane;
                        total = total.wrapping_add(values[index]); output[index] = total;
                    }
                } }
            }
            return Tensor::from_vec_i64(output, self.shape());
        } else {
            let values = real_values(self, output_dtype, "cumsum")?;
            let mut output = crate::typed_math::filled("cumsum", self.shape(), 0.0f64)?;
            if count != 0 {
                for group in 0..outer { for lane in 0..inner {
                    let mut total = 0.0f64;
                    for step in 0..width {
                        let position = if reverse { width - 1 - step } else { step };
                        let index = (group * width + position) * inner + lane;
                        total += values[index]; output[index] = rounded(total, output_dtype);
                    }
                } }
            }
            real_tensor(output, self.shape(), output_dtype, "cumsum")?
        };
        let input_shape = self.shape().to_vec();
        let input_dtype = self.dtype();
        let backward = move |g: &Tensor| {
            vec![crate::autograd::with_grad_enabled(false, || selected_vjp(g, &input_shape, input_dtype, dim, reverse)).expect("validated cumulative sum adjoint")]
        };
        // Preserve the original F32 inference barrier; it never claimed replay.
        if crate::capture::is_recording() { return Ok(output.record_fn(vec![self.clone()], backward)); }
        Ok(output.record_fn_forward(vec![self.clone()], ForwardOp::CumSum { dim, dtype, reverse }, backward))
    }
}
