//! Typed CPU rounded division, with separate floor_divide AD semantics.
use crate::{DType, Device, Error, Result, Tensor};
use crate::autograd::ForwardOp;
use crate::shape::broadcast_shapes;
use crate::typed_math::{copy_pointwise_layout, binary_pointwise_layout, pointwise_source, pointwise_values_f32, pointwise_values_f64};

#[derive(Clone, Copy, Debug)]
pub enum DivisionRounding { Floor, Trunc }

macro_rules! floor_value {
    ($name:ident, $float:ty) => {
        fn $name(a: $float, b: $float) -> $float {
            if b == 0.0 { return a / b; }
            let remainder = a % b;
            let mut quotient = (a - remainder) / b;
            if remainder != 0.0 && (remainder < 0.0) != (b < 0.0) { quotient -= 1.0; }
            if quotient == 0.0 { return (0.0 as $float).copysign(a / b); }
            let mut result = quotient.floor();
            if quotient - result > 0.5 { result += 1.0; }
            result
        }
    };
}
floor_value!(floor_f32, f32);
floor_value!(floor_f64, f64);

pub(crate) fn zero_like(input: &Tensor) -> Result<Tensor> { zero_with_dtype(input, input.dtype()) }

fn zero_with_dtype(input: &Tensor, dtype: DType) -> Result<Tensor> {
    let raw = match dtype {
        DType::F32 => Tensor::from_vec(crate::typed_math::filled("rounded_div backward", input.shape(), 0.0f32)?, input.shape())?,
        DType::F64 => crate::typed_math::full_f64(input.shape(), 0.0)?,
        got => return Err(Error::DtypeMismatch { op: "rounded_div backward", expected: DType::F32, got }),
    };
    Ok(Tensor::from_parts(raw.0.storage.clone(), input.shape().to_vec(), copy_pointwise_layout(input), 0, Device::Cpu, false, None))
}

pub(crate) fn zero_vjp(inputs: &[Tensor], gradient: &Tensor, required: Option<&[bool]>) -> Result<Vec<Option<Tensor>>> {
    inputs.iter().enumerate().map(|(index, input)| {
        if !matches!(input.dtype(), DType::F32 | DType::F64) || required.is_some_and(|mask| !mask[index]) { Ok(None) }
        else {
            let zero = zero_like(gradient)?;
            let reduced = crate::autograd::with_grad_enabled(false, || crate::tensor::unbroadcast(&zero, input.shape()));
            if reduced.dtype() == input.dtype() { Ok(Some(reduced)) } else { zero_with_dtype(&reduced, input.dtype()).map(Some) }
        }
    }).collect()
}

fn as_i64(input: &Tensor) -> Vec<i64> {
    match input.dtype() { DType::I64 => input.to_vec_i64(), DType::Bool => input.to_vec_bool().into_iter().map(i64::from).collect(), _ => unreachable!() }
}

impl Tensor {
    pub fn div_rounding(&self, other: &Tensor, mode: DivisionRounding) -> Result<Tensor> { self.rounded_division(other, mode, false) }
    pub fn floor_divide(&self, other: &Tensor) -> Result<Tensor> { self.rounded_division(other, DivisionRounding::Floor, true) }

    fn rounded_division(&self, other: &Tensor, mode: DivisionRounding, floor_api: bool) -> Result<Tensor> {
        let op = if floor_api { "floor_divide" } else { "div_rounding" };
        if crate::capture::is_recording() { return Err(Error::Unsupported { op, msg: "rounded division currently requires eager CPU execution".into() }); }
        if self.device() != other.device() { return Err(Error::DeviceMismatch { op, lhs: self.device(), rhs: other.device() }); }
        if self.device() != Device::Cpu { return Err(Error::Unsupported { op, msg: "rounded division currently requires CPU operands".into() }); }
        let dtype = crate::typed_math::result_type(self, other)?;
        if dtype == DType::Bool { return Err(Error::Unsupported { op, msg: "boolean rounded division is not supported".into() }); }
        let shape = broadcast_shapes(op, self.shape(), other.shape())?;
        let count = crate::typed_math::check_shape(op, &shape)?;
        if count != 0 { crate::typed_math::check_shape(op, self.shape())?; crate::typed_math::check_shape(op, other.shape())?; }
        let stride = binary_pointwise_layout(self, other, &shape, dtype);
        let (a, b) = (pointwise_source(self, &shape, &stride)?, pointwise_source(other, &shape, &stride)?);
        let raw = match dtype {
            DType::F32 => {
                let mut values = crate::typed_math::filled(op, &shape, 0.0f32)?;
                for ((value, x), y) in values.iter_mut().zip(pointwise_values_f32(&a)).zip(pointwise_values_f32(&b)) { *value = match mode { DivisionRounding::Floor => floor_f32(x, y), DivisionRounding::Trunc => (x / y).trunc() }; }
                Tensor::from_vec(values, &shape)?
            }
            DType::F64 => {
                let mut values = crate::typed_math::filled(op, &shape, 0.0f64)?;
                for ((value, x), y) in values.iter_mut().zip(pointwise_values_f64(&a)).zip(pointwise_values_f64(&b)) { *value = match mode { DivisionRounding::Floor => floor_f64(x, y), DivisionRounding::Trunc => (x / y).trunc() }; }
                Tensor::from_vec_f64(values, &shape)?
            }
            DType::I64 => {
                let mut values = crate::typed_math::filled(op, &shape, 0i64)?;
                for ((value, x), y) in values.iter_mut().zip(as_i64(&a)).zip(as_i64(&b)) {
                    if x == i64::MIN && y == -1 && matches!(mode, DivisionRounding::Floor) { *value = i64::MIN; continue; }
                    let q = x.checked_div(y).ok_or_else(|| Error::InvalidShape { op, msg: if y == 0 { "integer division by zero".into() } else { "integer division overflow".into() } })?;
                    *value = if matches!(mode, DivisionRounding::Floor) && x % y != 0 && (x < 0) != (y < 0) { q - 1 } else { q };
                }
                Tensor::from_vec_i64(values, &shape)?
            }
            _ => unreachable!(),
        };
        let output = Tensor::from_parts(raw.0.storage.clone(), shape, stride, 0, Device::Cpu, false, None);
        let forward = if floor_api { ForwardOp::FloorDivide } else { ForwardOp::RoundedDiv(mode) };
        let metadata: Vec<_> = [self, other].iter().map(|input| (input.shape().to_vec(), input.dtype())).collect();
        Ok(output.record_fn_forward_optional(vec![self.clone(), other.clone()], forward, move |_| {
            metadata.iter().map(|(shape, dtype)| match dtype {
                DType::F32 => Some(Tensor::from_vec(crate::typed_math::filled("div_rounding backward", shape, 0.0f32).unwrap(), shape).unwrap()),
                DType::F64 => Some(crate::typed_math::full_f64(shape, 0.0).unwrap()), _ => None,
            }).collect()
        }))
    }
}
