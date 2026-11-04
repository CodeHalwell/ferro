//! Elementwise select with torch.where semantics: out = if cond != 0 { a }
//! else { b }, broadcasting across all three operands. `cond` is a boolean or numeric mask
//! and receives no gradient; only `a` and `b` are differentiable inputs.

use crate::tensor::{unbroadcast, Tensor};
use crate::{DType, Device, Error, Result};
use crate::shape::{broadcast_shapes, checked_numel};
use crate::autograd::ForwardOp;

fn zero(dtype: DType) -> Result<Tensor> {
    if dtype == DType::F64 { Tensor::from_vec_f64(vec![0.0], &[]) }
    else { Ok(Tensor::scalar(0.0)) }
}

pub(crate) fn graph_vjp(inputs: &[Tensor], g: &Tensor) -> Result<Vec<Option<Tensor>>> {
    let reduce = |value: Tensor, shape: &[usize]| -> Result<Tensor> {
        if value.numel() == 0 { return crate::tensor::sum_empty_to_shape(&value, shape); }
        let mut value = value;
        while value.ndim() > shape.len() { value = value.sum_dim(0, false)?; }
        for (axis, &size) in shape.iter().enumerate() {
            if size == 1 && value.shape()[axis] != 1 { value = value.sum_dim(axis, true)?; }
        }
        Ok(value)
    };
    let zeros = zero(g.dtype())?;
    let left = Tensor::where_cond(&inputs[0], g, &zeros)?;
    let right = Tensor::where_cond(&inputs[0], &zeros, g)?;
    Ok(vec![None, Some(reduce(left, inputs[1].shape())?), Some(reduce(right, inputs[2].shape())?)])
}

pub(crate) fn jvp(inputs: &[Tensor], output: &Tensor, tangents: &[Option<Tensor>]) -> Result<Tensor> {
    let tangent = |index: usize| -> Result<Tensor> {
        match &tangents[index] {
            Some(value) => Ok(value.clone()),
            None => zero(inputs[index].dtype())?.expand(inputs[index].shape()),
        }
    };
    let left = tangent(1)?;
    let right = tangent(2)?;
    let dtype = crate::typed_math::result_type(&left, &right)?;
    let prepare = |value: Tensor| -> Result<Tensor> {
        // Empty output must not materialize a nonempty broadcast operand.
        let value = if output.numel() == 0 { value.expand(output.shape())? } else { value };
        value.to_dtype_autograd(dtype, false)
    };
    Tensor::where_cond(&inputs[0], &prepare(left)?, &prepare(right)?)
}

// The same typed kernel serves the primal and the ordinary masked adjoints.
// CPU iteration follows condition/left/right order; device fallback is an
// explicit host result and retains the existing contiguous fallback policy.
fn select(cond: &Tensor, a: &Tensor, b: &Tensor, shape: &[usize], planned: Option<&[usize]>) -> Result<Tensor> {
    let inputs = [cond, a, b];
    let resident = inputs.iter().all(|input| input.device() == Device::Cpu);
    let stride = if resident {
        planned.map(<[usize]>::to_vec).unwrap_or_else(|| crate::typed_math::iterator_pointwise_layout(&inputs, &inputs.map(|input| input.strides().to_vec()), shape))
    } else { crate::shape::default_strides(shape) };
    let prepare = |input: &Tensor| if resident { crate::typed_math::pointwise_source(input, shape, &stride) } else { input.broadcast_to(shape) };
    let keep = prepare(cond)?.to_vec_bool();
    let av = prepare(a)?;
    let bv = prepare(b)?;
    macro_rules! select {
        ($a:expr, $b:expr, $ctor:ident) => {
            Tensor::$ctor($a.into_iter().zip($b).zip(&keep).map(|((a,b), &k)| if k { a } else { b }).collect(), shape)?
        };
    }
    let raw = match a.dtype() {
        DType::F32 => select!(av.to_vec(), bv.to_vec(), from_vec),
        DType::F64 => select!(av.to_vec_f64(), bv.to_vec_f64(), from_vec_f64),
        DType::I64 => select!(av.to_vec_i64(), bv.to_vec_i64(), from_vec_i64),
        DType::Bool => select!(av.to_vec_bool(), bv.to_vec_bool(), from_vec_bool),
        DType::F16 => select!(av.to_vec_f16_bits()?, bv.to_vec_f16_bits()?, from_vec_f16_bits),
        DType::BF16 => select!(av.to_vec_bf16_bits()?, bv.to_vec_bf16_bits()?, from_vec_bf16_bits),
    };
    Ok(Tensor::from_parts(raw.0.storage.clone(), shape.to_vec(), stride, 0, Device::Cpu, false, None))
}

impl Tensor {
    pub fn where_cond(cond: &Tensor, a: &Tensor, b: &Tensor) -> crate::Result<Tensor> {
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op: "where_cond", msg: "inference capture is not supported".into() });
        }
        for t in [a, b] {
            if cond.device() != t.device() {
                return Err(Error::DeviceMismatch { op: "where_cond", lhs: cond.device(), rhs: t.device() });
            }
        }
        let shape = broadcast_shapes("where_cond", &broadcast_shapes("where_cond", a.shape(), b.shape())?, cond.shape())?;
        let dtype = if a.dtype() == b.dtype() { a.dtype() } else { crate::typed_math::result_type(a, b)? };
        if dtype == DType::F64 && a.device() != Device::Cpu {
            return Err(Error::Unsupported { op: "where_cond", msg: "float64 selection requires CPU storage".into() });
        }
        let bytes = match dtype { DType::F64 | DType::I64 => 8, DType::F32 => 4, DType::Bool => 1, DType::F16 | DType::BF16 => 2 };
        let count = checked_numel("where_cond", &shape)?;
        let read_bytes = bytes.max(if cond.dtype() == DType::Bool { 1 } else { 8 });
        if count > isize::MAX as usize / read_bytes {
            return Err(Error::InvalidShape { op: "where_cond", msg: "selection exceeds allocation capacity".into() });
        }
        // Plan from original metadata before empty expansion avoids a large
        // cast while retaining the original iterator's empty-axis ordering.
        let planned = if a.dtype() != b.dtype() {
            let strides = [cond.strides().to_vec(),
                if a.dtype() == dtype { a.strides().to_vec() } else { crate::typed_math::copy_pointwise_layout(a) },
                if b.dtype() == dtype { b.strides().to_vec() } else { crate::typed_math::copy_pointwise_layout(b) }];
            Some(crate::typed_math::iterator_pointwise_layout(&[cond, a, b], &strides, &shape))
        } else { None };
        let promoted = if planned.is_some() {
            let prepare = |value: &Tensor| -> Result<Tensor> {
                let value = if count == 0 { value.expand(&shape)? } else { value.clone() };
                value.to_dtype_autograd(dtype, false)
            };
            Some((prepare(a)?, prepare(b)?))
        } else { None };
        let (a, b) = promoted.as_ref().map(|(a, b)| (a, b)).unwrap_or((a, b));
        let out = select(cond, a, b, &shape, planned.as_deref())?;
        if !matches!(dtype, DType::F32 | DType::F64) { return Ok(out); }
        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        // A detached condition handle cannot retain the output graph. Its
        // storage version is validated by the existing Where descriptor.
        let condition = cond.detach();
        Ok(out.record_fn_forward_optional(vec![cond.detach(), a.clone(), b.clone()], ForwardOp::WhereConditional, move |g| {
            let zeros = zero(g.dtype()).expect("validated Where gradient dtype");
            let ga = select(&condition, g, &zeros, &shape, None).expect("validated Where left adjoint");
            let gb = select(&condition, &zeros, g, &shape, None).expect("validated Where right adjoint");
            vec![None, Some(unbroadcast(&ga, &a_shape)), Some(unbroadcast(&gb, &b_shape))]
        }))
    }
}

#[cfg(test)]
#[path = "where_layout_tests.rs"]
mod layout_tests;
