//! CPU add/sub with an unrounded scalar coefficient and fused floating primal.
use crate::{DType, Device, Error, Result, Tensor};
use crate::autograd::ForwardOp;
use crate::shape::broadcast_shapes;
use crate::tensor::unbroadcast;

/// The original scalar category is part of add/sub's conversion contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Alpha { Bool(bool), I64(i64), U64(u64), F64(f64) }

impl Alpha {
    fn negated_primal(self) -> Self {
        match self {
            Self::Bool(_) => unreachable!("boolean subtraction is rejected"),
            Self::I64(value) => Self::I64(value.wrapping_neg()),
            Self::U64(value) => Self::I64((value as i64).wrapping_neg()),
            Self::F64(value) => Self::F64(-value),
        }
    }

    fn f32(self) -> f32 {
        match self { Self::Bool(value) => u8::from(value) as f32, Self::I64(value) => value as f32, Self::U64(value) => value as f32, Self::F64(value) => value as f32 }
    }

    fn f64(self) -> f64 {
        match self { Self::Bool(value) => u8::from(value) as f64, Self::I64(value) => value as f64, Self::U64(value) => value as f64, Self::F64(value) => value }
    }

    fn validate(self, op: &'static str, dtype: DType, subtract: bool) -> Result<()> {
        let invalid = |msg: &str| Error::InvalidShape { op, msg: msg.into() };
        if matches!(self, Self::Bool(_)) && dtype != DType::Bool {
            return Err(invalid("boolean alpha is supported only for boolean results"));
        }
        if matches!(self, Self::F64(_)) && matches!(dtype, DType::Bool | DType::I64) {
            return Err(invalid("integral results require an integer alpha"));
        }
        if dtype == DType::I64 && !subtract && matches!(self, Self::U64(value) if value > i64::MAX as u64) {
            return Err(invalid("alpha cannot be converted to int64 without overflow"));
        }
        if dtype == DType::F32 && matches!(self, Self::F64(value) if value.is_finite() && value.abs() > f32::MAX as f64) {
            return Err(invalid("alpha cannot be converted to float32 without overflow"));
        }
        Ok(())
    }

    pub(crate) fn validate_ad(self) -> Result<()> {
        if matches!(self, Self::U64(value) if value > i64::MAX as u64) {
            return Err(Error::InvalidShape { op: "add_alpha derivative", msg: "alpha cannot be converted to int64 without overflow".into() });
        }
        Ok(())
    }

    pub(crate) fn scale(self, gradient: &Tensor) -> Result<Tensor> {
        self.validate_ad()?;
        if matches!(self, Self::I64(1) | Self::U64(1) | Self::F64(1.0)) { return Ok(gradient.clone()); }
        let scalar = match gradient.dtype() {
            DType::F32 => Tensor::scalar(self.f32()),
            DType::F64 => Tensor::from_vec_f64(vec![self.f64()], &[])?,
            dtype => return Err(Error::DtypeMismatch { op: "add_alpha backward", expected: DType::F32, got: dtype }),
        };
        gradient.mul(&scalar)
    }
}

// The same Op can request only a subset of its input adjoints. In particular,
// Torch permits unsigned-alpha forward and the left derivative even when its
// scalar conversion rejects the right derivative.
pub(crate) fn selected_vjp(inputs: &[Tensor], gradient: &Tensor, alpha: Alpha, subtract: bool, required: Option<&[bool]>, create_graph: bool) -> Result<Vec<Option<Tensor>>> {
    let needed = |index| required.is_none_or(|mask| mask[index]);
    if needed(1) { alpha.validate_ad()?; }
    let reduce = |value: &Tensor, input: &Tensor| -> Result<Tensor> {
        let reduced = if value.numel() == 0 {
            crate::tensor::sum_empty_to_shape(value, input.shape())?
        } else if create_graph {
            let mut value = value.clone();
            while value.ndim() > input.ndim() { value = value.sum_dim(0, false)?; }
            for (axis, &size) in input.shape().iter().enumerate() {
                if size == 1 && value.shape()[axis] != 1 { value = value.sum_dim(axis, true)?; }
            }
            value
        } else { unbroadcast(value, input.shape()) };
        if create_graph { reduced.to_dtype_autograd(input.dtype(), false) }
        else { Ok(if reduced.dtype() == input.dtype() { reduced } else { reduced.to_dtype(input.dtype()) }) }
    };
    let left = if needed(0) { Some(reduce(gradient, &inputs[0])?) } else { None };
    let right = if needed(1) {
        let signed = if subtract { gradient.neg() } else { gradient.clone() };
        Some(reduce(&alpha.scale(&signed)?, &inputs[1])?)
    } else { None };
    Ok(vec![left, right])
}

impl Tensor {
    /// Fused CPU `self + alpha * other`, promoting operands before scalar
    /// conversion. Integer results wrap; inference capture is unsupported.
    pub fn add_alpha(&self, other: &Tensor, alpha: Alpha) -> Result<Tensor> {
        self.alpha_binary(other, alpha, false)
    }

    /// Fused CPU subtraction. Torch's I64MIN scalar negation wraps in the
    /// primal, while its AD coefficient is positive; this special scalar does
    /// not satisfy a finite-difference derivative of the observed primal.
    pub fn sub_alpha(&self, other: &Tensor, alpha: Alpha) -> Result<Tensor> {
        self.alpha_binary(other, alpha, true)
    }

    fn alpha_binary(&self, other: &Tensor, alpha: Alpha, subtract: bool) -> Result<Tensor> {
        let op = if subtract { "sub_alpha" } else { "add_alpha" };
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op, msg: "alpha arithmetic currently requires eager CPU execution".into() });
        }
        if self.device() != other.device() { return Err(Error::DeviceMismatch { op, lhs: self.device(), rhs: other.device() }); }
        if self.device() != Device::Cpu { return Err(Error::Unsupported { op, msg: "alpha arithmetic currently requires CPU operands".into() }); }
        if subtract && (self.dtype() == DType::Bool || other.dtype() == DType::Bool) {
            return Err(Error::Unsupported { op, msg: "subtraction involving boolean tensors is unsupported".into() });
        }
        let dtype = crate::typed_math::result_type(self, other)?;
        alpha.validate(op, dtype, subtract)?;
        let shape = broadcast_shapes(op, self.shape(), other.shape())?;
        let count = crate::typed_math::check_shape(op, &shape)?;
        if count != 0 {
            crate::typed_math::check_shape(op, self.shape())?;
            crate::typed_math::check_shape(op, other.shape())?;
        }
        // Empty broadcast results need neither operand's values nor a cast
        // workspace. Keep the original edges and cast their adjoints instead.
        let a = if count == 0 { self.clone() } else { crate::typed_math::promoted_cast(self, dtype)? };
        let b = if count == 0 { other.clone() } else { crate::typed_math::promoted_cast(other, dtype)? };
        let coefficient = if subtract { alpha.negated_primal() } else { alpha };
        let av = a.broadcast_to(&shape)?;
        let bv = b.broadcast_to(&shape)?;
        let output = match dtype {
            DType::F32 => {
                let mut values = crate::typed_math::filled(op, &shape, 0.0f32)?;
                let scale = coefficient.f32();
                for ((output, x), y) in values.iter_mut().zip(av.to_vec()).zip(bv.to_vec()) { *output = y.mul_add(scale, x); }
                Tensor::from_vec(values, &shape)?
            }
            DType::F64 => {
                let mut values = crate::typed_math::filled(op, &shape, 0.0f64)?;
                let scale = coefficient.f64();
                for ((output, x), y) in values.iter_mut().zip(av.to_vec_f64()).zip(bv.to_vec_f64()) { *output = y.mul_add(scale, x); }
                Tensor::from_vec_f64(values, &shape)?
            }
            DType::I64 => {
                let scale = match coefficient { Alpha::I64(value) => value, Alpha::U64(value) => value as i64, _ => unreachable!() };
                let mut values = crate::typed_math::filled(op, &shape, 0i64)?;
                for ((output, x), y) in values.iter_mut().zip(av.to_vec_i64()).zip(bv.to_vec_i64()) { *output = x.wrapping_add(y.wrapping_mul(scale)); }
                return Tensor::from_vec_i64(values, &shape);
            }
            DType::Bool => {
                let scale = match coefficient { Alpha::Bool(value) => value, Alpha::I64(value) => value != 0, Alpha::U64(value) => value != 0, _ => unreachable!() };
                let mut values = crate::typed_math::filled(op, &shape, false)?;
                for ((output, x), y) in values.iter_mut().zip(av.to_vec_bool()).zip(bv.to_vec_bool()) { *output = x || (y && scale); }
                return Tensor::from_vec_bool(values, &shape);
            }
            _ => unreachable!("result_type accepts only supported CPU dtypes"),
        };
        let (left, right) = (a.shape().to_vec(), b.shape().to_vec());
        let (left_dtype, right_dtype) = (a.dtype(), b.dtype());
        Ok(output.record_fn_forward(vec![a, b], ForwardOp::AddAlpha { alpha, subtract }, move |gradient| {
            let signed = if subtract { gradient.neg() } else { gradient.clone() };
            let right_gradient = alpha.scale(&signed).expect("validated floating alpha");
            let cast = |value: Tensor, dtype| if value.dtype() == dtype { value } else { value.to_dtype(dtype) };
            vec![cast(unbroadcast(gradient, &left), left_dtype), cast(unbroadcast(&right_gradient, &right), right_dtype)]
        }))
    }
}
