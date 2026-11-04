//! CPU typed kernels. F32 keeps its backend dispatch; these paths never narrow
//! values to f32 and deliberately decline inference capture.
use crate::{DType, Device, Error, Result, Tensor};
use crate::dispatch::{BinaryKind, UnaryKind};
use crate::shape::{broadcast_shapes, checked_numel, default_strides};

pub(crate) fn check_shape(op: &'static str, shape: &[usize]) -> Result<usize> {
    let count = checked_numel(op, shape)?;
    if count > isize::MAX as usize / 8 {
        return Err(Error::InvalidShape { op, msg: "typed materialization exceeds allocation capacity".into() });
    }
    Ok(count)
}

pub(crate) fn check(op: &'static str, inputs: &[&Tensor], dtype: DType) -> Result<()> {
    if crate::capture::is_recording() {
        return Err(Error::Unsupported { op, msg: "typed CPU computation does not support inference capture".into() });
    }
    for t in inputs {
        if t.dtype() != dtype { return Err(Error::DtypeMismatch { op, expected: dtype, got: t.dtype() }); }
        if t.device() != Device::Cpu {
            return Err(Error::Unsupported { op, msg: "typed computation is supported on CPU only".into() });
        }
        check_shape(op, t.shape())?;
    }
    Ok(())
}

pub(crate) fn filled<T: Clone>(op: &'static str, shape: &[usize], value: T) -> Result<Vec<T>> {
    let n = checked_numel(op, shape)?;
    if n > isize::MAX as usize / std::mem::size_of::<T>().max(1) {
        return Err(Error::InvalidShape { op, msg: "output exceeds allocation capacity".into() });
    }
    let mut values = Vec::new();
    values.try_reserve_exact(n).map_err(|_| Error::InvalidShape { op, msg: "cannot allocate output".into() })?;
    values.resize(n, value);
    Ok(values)
}

pub(crate) fn full_f64(shape: &[usize], value: f64) -> Result<Tensor> {
    Tensor::from_vec_f64(filled("full_f64", shape, value)?, shape)
}

fn category(dtype: DType) -> Option<u8> {
    match dtype { DType::Bool => Some(0), DType::I64 => Some(1), DType::F32 | DType::F64 => Some(2), _ => None }
}

/// Torch-style tensor promotion: dimensional operands take precedence over
/// zero-dimensional operands within one category, while floats outrank ints.
pub(crate) fn result_type(a: &Tensor, b: &Tensor) -> Result<DType> {
    let (ad, bd) = (a.dtype(), b.dtype());
    let ac = category(ad).ok_or_else(|| Error::Unsupported { op: "promote", msg: "half storage has no arithmetic promotion rule".into() })?;
    let bc = category(bd).ok_or_else(|| Error::Unsupported { op: "promote", msg: "half storage has no arithmetic promotion rule".into() })?;
    Ok(if ac != bc { if ac > bc { ad } else { bd } }
       else if a.ndim() == 0 && b.ndim() != 0 { bd }
       else if b.ndim() == 0 && a.ndim() != 0 { ad }
       else if ad == DType::F64 || bd == DType::F64 { DType::F64 }
       else { ad })
}

pub(crate) fn promoted_binary(op: &'static str, a: &Tensor, b: &Tensor, kind: BinaryKind) -> Result<Option<Tensor>> {
    let (ad, bd) = (a.dtype(), b.dtype());
    if ad == bd && matches!(ad, DType::F32 | DType::F64) { return Ok(None); }
    if ad == bd && ad == DType::I64 && !matches!(kind, BinaryKind::Div) { return Ok(None); }
    if category(ad).is_none() || category(bd).is_none() { return Ok(None); }
    if a.device() != b.device() { return Err(Error::DeviceMismatch { op, lhs: a.device(), rhs: b.device() }); }
    if a.device() != Device::Cpu {
        return Err(Error::Unsupported { op, msg: "typed promotion currently requires CPU operands".into() });
    }
    if matches!(kind, BinaryKind::Sub) && (ad == DType::Bool || bd == DType::Bool) {
        return Err(Error::Unsupported { op, msg: "subtraction involving boolean tensors is unsupported".into() });
    }
    if ad == DType::Bool && bd == DType::Bool {
        match kind {
            BinaryKind::Add => return a.logical_or(b).map(Some),
            BinaryKind::Mul => return a.logical_and(b).map(Some),
            _ => {},
        }
    }
    let mut dtype = result_type(a, b)?;
    if matches!(kind, BinaryKind::Div) && matches!(dtype, DType::I64 | DType::Bool) { dtype = DType::F32; }
    // Efficient-zero addition copies the broadcast ordinary operand. Expand
    // before its conversion so empty/gapped copies keep that original layout.
    let (a, b) = if matches!(kind, BinaryKind::Add | BinaryKind::Sub) && a.is_structural_zero() != b.is_structural_zero() {
        let shape = broadcast_shapes(op, a.shape(), b.shape())?;
        (promoted_cast(&a.expand(&shape)?, dtype)?, promoted_cast(&b.expand(&shape)?, dtype)?)
    } else { (promoted_cast(a, dtype)?, promoted_cast(b, dtype)?) };
    let out = match kind { BinaryKind::Add => a.add(&b), BinaryKind::Sub => a.sub(&b), BinaryKind::Mul => a.mul(&b), BinaryKind::Div => a.div(&b), BinaryKind::ReluBackward => Err(Error::Unsupported { op, msg: "relu backward does not use arithmetic promotion".into() }) }?;
    Ok(Some(out))
}

pub(crate) fn promoted_cast(input: &Tensor, dtype: DType) -> Result<Tensor> {
    if !input.is_structural_zero() || input.dtype() == dtype { return input.to_dtype_autograd(dtype, false); }
    if crate::capture::is_recording() {
        return Err(Error::Unsupported { op: "promote", msg: "structural-zero arithmetic requires eager CPU execution".into() });
    }
    let original = input.dtype();
    let out = Tensor::structural_zero(input.shape(), dtype, input.device())?;
    Ok(out.record_fn_forward(vec![input.clone()], crate::autograd::ForwardOp::Cast(dtype), move |g| vec![g.to_dtype(original)]))
}

pub(crate) fn binary(op: &'static str, a: &Tensor, b: &Tensor, kind: BinaryKind) -> Result<Tensor> {
    check(op, &[a, b], a.dtype())?;
    if let Some(output) = crate::tensor::raw_structural_binary(op, a, b, kind)? { return Ok(output); }
    let shape = broadcast_shapes(op, a.shape(), b.shape())?;
    match a.dtype() {
        DType::F64 => {
            let mut out = filled(op, &shape, 0.0f64)?;
            let stride = binary_pointwise_layout(a, b, &shape, DType::F64);
            let av = pointwise_source(a, &shape, &stride)?.to_vec_f64();
            let bv = pointwise_source(b, &shape, &stride)?.to_vec_f64();
            for ((o, x), y) in out.iter_mut().zip(av).zip(bv) {
                *o = match kind { BinaryKind::Add => x + y, BinaryKind::Sub => x - y, BinaryKind::Mul => x * y, BinaryKind::Div => x / y, BinaryKind::ReluBackward => if y <= 0.0 { 0.0 } else { x } };
            }
            let raw = Tensor::from_vec_f64(out, &shape)?;
            Ok(Tensor::from_parts(raw.0.storage.clone(), shape, stride, 0, Device::Cpu, false, None))
        }
        DType::I64 => {
            if matches!(kind, BinaryKind::Div | BinaryKind::ReluBackward) {
                return Err(Error::Unsupported { op, msg: "integer true division requires an explicit floating-point cast".into() });
            }
            let mut out = filled(op, &shape, 0i64)?;
            let av = a.broadcast_to(&shape)?.to_vec_i64();
            let bv = b.broadcast_to(&shape)?.to_vec_i64();
            for ((o, x), y) in out.iter_mut().zip(av).zip(bv) {
                *o = match kind { BinaryKind::Add => x.wrapping_add(y), BinaryKind::Sub => x.wrapping_sub(y), BinaryKind::Mul => x.wrapping_mul(y), BinaryKind::Div | BinaryKind::ReluBackward => unreachable!() };
            }
            Tensor::from_vec_i64(out, &shape)
        }
        _ => Err(Error::DtypeMismatch { op, expected: DType::F64, got: a.dtype() }),
    }
}

pub(crate) fn unary(a: &Tensor, kind: UnaryKind) -> Result<Tensor> {
    check("unary", &[a], a.dtype())?;
    if a.dtype() == DType::I64 && matches!(kind, UnaryKind::Neg | UnaryKind::Sign) {
        let stride = unary_pointwise_layout(a);
        let physical = pointwise_source(a, a.shape(), &stride)?;
        let raw = Tensor::from_vec_i64(physical.to_vec_i64().into_iter().map(|x| if matches!(kind, UnaryKind::Sign) { x.signum() } else { x.wrapping_neg() }).collect(), a.shape())?;
        return Ok(Tensor::from_parts(raw.0.storage.clone(), a.shape().to_vec(), stride, 0, Device::Cpu, false, None));
    }
    if a.dtype() != DType::F64 { return Err(Error::DtypeMismatch { op: "unary", expected: DType::F64, got: a.dtype() }); }
    if matches!(kind, UnaryKind::Gelu | UnaryKind::GeluErf | UnaryKind::GeluErfGrad | UnaryKind::Silu) {
        return Err(Error::Unsupported { op: "unary", msg: "this activation has no float64 kernel".into() });
    }
    let stride = unary_pointwise_layout(a);
    let physical = pointwise_source(a, a.shape(), &stride)?;
    let values = physical.to_vec_f64().into_iter().map(|x| match kind {
        UnaryKind::Neg => -x,
        UnaryKind::Relu => if x.is_nan() { x } else { x.max(0.0) },
        UnaryKind::Exp => x.exp(),
        UnaryKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UnaryKind::Tanh => x.tanh(),
        UnaryKind::Sqrt => x.sqrt(),
        UnaryKind::Abs => x.abs(),
        UnaryKind::Sign => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        UnaryKind::Log => x.ln(),
        UnaryKind::Powf(p) => if p == 0.5 { x.sqrt() } else if p == -0.5 { x.sqrt().recip() } else if p == -1.0 { x.recip() } else if p == -2.0 { (x * x).recip() } else if p == 2.0 { x * x } else if p == 3.0 { (x * x) * x } else { x.powf(p as f64) },
        UnaryKind::Clamp { min, max } => if x.is_nan() || min.is_nan() || max.is_nan() { f64::NAN } else { let lower = if x < min as f64 { min as f64 } else { x }; if lower > max as f64 { max as f64 } else { lower } },
        UnaryKind::Gtz => if !(x <= 0.0) { 1.0 } else { 0.0 },
        _ => unreachable!(),
    }).collect();
    let raw = Tensor::from_vec_f64(values, a.shape())?;
    Ok(Tensor::from_parts(raw.0.storage.clone(), a.shape().to_vec(), stride, 0, Device::Cpu, false, None))
}

pub(crate) fn unary_grad(
    a: &Tensor, op: &'static str,
    forward: impl Fn(f64) -> f64,
    backward: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
) -> Result<Tensor> {
    unary_grad_tensor(a, op, forward, false, move |g, x| {
        let (g, x) = (numeric_zero_operand(g)?, numeric_zero_operand(x)?);
        pointwise_binary_real(op, &g, &x, |_,_| unreachable!("validated float64 backward"), &backward)
    })
}

// Unsupported efficient-zero kernels receive ordinary numeric zeros, with
// zero strides deferring iteration order to their concrete operands.
fn numeric_zero_operand(input: &Tensor) -> Result<Tensor> {
    if input.is_structural_zero() { full_f64(&[], 0.0)?.expand(input.shape()) }
    else { Ok(input.clone()) }
}

pub(crate) fn unary_grad_tensor(
    a: &Tensor, op: &'static str,
    forward: impl Fn(f64) -> f64,
    saves_output: bool,
    backward: impl Fn(&Tensor, &Tensor) -> Result<Tensor> + Send + Sync + 'static,
) -> Result<Tensor> {
    check(op, &[a], DType::F64)?;
    let stride = unary_pointwise_layout(a);
    let physical = pointwise_source(a, a.shape(), &stride)?;
    let raw = Tensor::from_vec_f64(physical.to_vec_f64().into_iter().map(forward).collect(), a.shape())?;
    let out = Tensor::from_parts(raw.0.storage.clone(), a.shape().to_vec(), stride, 0, Device::Cpu, false, None);
    if !a.requires_grad() && !crate::forward_ad::is_active() { return Ok(out); }
    let saved = if saves_output { out.detach_copy_layout()? } else { a.detach() };
    let callback = move |g: &Tensor| vec![backward(g, &saved).expect("validated float64 unary backward")];
    Ok(if saves_output { out.record_fn_saved(vec![a.clone()], &[], true, callback) }
       else { out.record_fn(vec![a.clone()], callback) })
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RoundingKind { Floor, Ceil, Trunc, Round(Option<i64>) }

impl RoundingKind {
    pub(crate) fn name(self) -> &'static str {
        match self { Self::Floor => "floor", Self::Ceil => "ceil", Self::Trunc => "trunc", Self::Round(_) => "round" }
    }

    fn f32(self, value: f32) -> f32 {
        match self {
            Self::Floor => value.floor(), Self::Ceil => value.ceil(), Self::Trunc => value.trunc(),
            Self::Round(None | Some(0)) => value.round_ties_even(),
            Self::Round(Some(decimals)) => {
                let scale = 10.0f64.powf(decimals.unsigned_abs() as f64) as f32;
                if decimals < 0 { (value / scale).round_ties_even() * scale }
                else { (value * scale).round_ties_even() / scale }
            }
        }
    }

    fn f64(self, value: f64) -> f64 {
        match self {
            Self::Floor => value.floor(), Self::Ceil => value.ceil(), Self::Trunc => value.trunc(),
            Self::Round(None | Some(0)) => value.round_ties_even(),
            Self::Round(Some(decimals)) => {
                let scale = 10.0f64.powf(decimals.unsigned_abs() as f64);
                if decimals < 0 { (value / scale).round_ties_even() * scale }
                else { (value * scale).round_ties_even() / scale }
            }
        }
    }
}

fn pointwise_layout(input: &Tensor) -> (Tensor, Vec<usize>) {
    let mut axes = (0..input.ndim()).rev().collect::<Vec<_>>();
    if input.numel() != 0 && !input.is_contiguous() {
        let dense_order = |axes: &[usize]| {
            let mut expected = 1;
            axes.iter().all(|&axis| {
                if input.shape()[axis] == 1 { return true; }
                if input.strides()[axis] != expected { return false; }
                expected *= input.shape()[axis];
                true
            })
        };
        let channels_last = input.ndim() == 4 && dense_order(&[1, 3, 2, 0]);
        let mut sorted = axes.clone();
        sorted.sort_by_key(|&axis| input.strides()[axis]);
        if channels_last || dense_order(&sorted) {
            let stride = if channels_last {
                let mut stride = vec![0; 4];
                let mut expected = 1;
                for axis in [1, 3, 2, 0] { stride[axis] = expected; expected *= input.shape()[axis].max(1); }
                stride
            } else { input.strides().to_vec() };
            let source = Tensor::from_parts(input.0.storage.clone(), vec![input.numel()], vec![1], input.storage_offset(), Device::Cpu, false, None);
            return (source, stride);
        }
        // A single-operand TensorIterator keeps ambiguous broadcast axes in
        // logical order, and orders concrete axes by their physical strides.
        for index in 1..axes.len() {
            let mut current = index;
            for previous in (0..index).rev() {
                let (left, right) = (axes[previous], axes[current]);
                let (ls, rs) = (input.strides()[left], input.strides()[right]);
                if ls == 0 || rs == 0 { continue; }
                if ls < rs { break; }
                if ls > rs || input.shape()[left] > input.shape()[right] {
                    axes.swap(previous, current);
                    current = previous;
                }
            }
        }
    }
    let mut stride = vec![0; axes.len()];
    let mut expected = 1;
    for &axis in &axes { stride[axis] = expected; expected *= input.shape()[axis].max(1); }
    axes.reverse();
    let source = Tensor::from_parts(input.0.storage.clone(), axes.iter().map(|&axis| input.shape()[axis]).collect(),
        axes.iter().map(|&axis| input.strides()[axis]).collect(), input.storage_offset(), Device::Cpu, false, None);
    (source, stride)
}

pub(crate) fn pointwise_real(input: &Tensor, op: &'static str, f32_fn: impl Fn(f32) -> f32, f64_fn: impl Fn(f64) -> f64) -> Result<Tensor> {
    check_shape(op, input.shape())?;
    if input.dtype() == DType::F32 && input.device() != Device::Cpu {
        return crate::tensor::raw_binary(op, input, input, |value, _| f32_fn(value))?.to_device(input.device());
    }
    if input.dtype() == DType::F64 { check(op, &[input], DType::F64)?; }
    let (source, stride) = pointwise_layout(input);
    let output = match input.dtype() {
        DType::F32 => Tensor::from_vec(source.to_vec().into_iter().map(f32_fn).collect(), source.shape())?,
        DType::F64 => {
            Tensor::from_vec_f64(source.to_vec_f64().into_iter().map(f64_fn).collect(), source.shape())?
        }
        got => return Err(Error::DtypeMismatch { op, expected: DType::F32, got }),
    };
    Ok(Tensor::from_parts(output.0.storage.clone(), input.shape().to_vec(), stride, 0, Device::Cpu, false, None))
}

pub(crate) fn rounding(input: &Tensor, kind: RoundingKind) -> Result<Tensor> {
    if input.dtype() == DType::I64 && !matches!(kind, RoundingKind::Round(Some(_))) {
        check(kind.name(), &[input], DType::I64)?;
        let (source, stride) = pointwise_layout(input);
        let output = Tensor::from_vec_i64(source.to_vec_i64(), source.shape())?;
        return Ok(Tensor::from_parts(output.0.storage.clone(), input.shape().to_vec(), stride, 0, Device::Cpu, false, None));
    }
    let out = pointwise_real(input, kind.name(), |value| kind.f32(value), |value| kind.f64(value))?;
    Ok(out.record_fn_forward(vec![input.clone()], crate::autograd::ForwardOp::Rounding(kind), |gradient| {
        vec![if gradient.dtype() == DType::F64 { full_f64(gradient.shape(), 0.0).unwrap() }
             else { Tensor::full_on(gradient.shape(), 0.0, gradient.device()).unwrap() }]
    }))
}

pub(crate) fn sum(t: &Tensor, dim: Option<usize>, keepdim: bool) -> Result<Tensor> {
    check("sum", &[t], t.dtype())?;
    let mut shape = t.shape().to_vec();
    let (outer, axis, inner) = if let Some(d) = dim {
        if d >= shape.len() { return Err(Error::InvalidShape { op: "sum", msg: "axis out of range".into() }); }
        let sizes = (shape[..d].iter().product::<usize>(), shape[d], shape[d + 1..].iter().product::<usize>());
        if keepdim { shape[d] = 1; } else { shape.remove(d); }
        sizes
    } else { shape.clear(); (1, t.numel(), 1) };
    macro_rules! reduce {
        ($data:expr, $zero:expr, $add:expr, $ctor:ident) => {{
            let mut out = filled("sum", &shape, $zero)?;
            if !out.is_empty() && axis != 0 {
                let values = $data;
                for o in 0..outer { for a in 0..axis { for i in 0..inner {
                    out[o * inner + i] = $add(out[o * inner + i], values[(o * axis + a) * inner + i]);
                } } }
            }
            Tensor::$ctor(out, &shape)
        }};
    }
    match t.dtype() {
        DType::F64 => reduce!(t.to_vec_f64(), 0.0f64, |a, b| a + b, from_vec_f64),
        DType::I64 => reduce!(t.to_vec_i64(), 0i64, i64::wrapping_add, from_vec_i64),
        _ => Err(Error::DtypeMismatch { op: "sum", expected: DType::F64, got: t.dtype() }),
    }
}

pub(crate) fn matmul(a: &Tensor, b: &Tensor, batched: bool) -> Result<Tensor> {
    let op = if batched { "bmm" } else { "matmul" };
    check(op, &[a, b], DType::F64)?;
    let rank = if batched { 3 } else { 2 };
    if a.ndim() != rank || b.ndim() != rank {
        return Err(Error::InvalidShape { op, msg: format!("inputs must have rank {rank}") });
    }
    let (batch, m, k, n) = (if batched { a.shape()[0] } else { 1 }, a.shape()[rank - 2], a.shape()[rank - 1], b.shape()[rank - 1]);
    if b.shape()[rank - 2] != k || (batched && b.shape()[0] != batch) {
        return Err(Error::ShapeMismatch { op, lhs: a.shape().to_vec(), rhs: b.shape().to_vec() });
    }
    let shape = if batched { vec![batch, m, n] } else { vec![m, n] };
    let mut out = filled(op, &shape, 0.0f64)?;
    if !out.is_empty() && k != 0 {
        let av = a.to_vec_f64();
        let bv = b.to_vec_f64();
        for bi in 0..batch { for i in 0..m { for p in 0..k { for j in 0..n {
            out[(bi * m + i) * n + j] += av[(bi * m + i) * k + p] * bv[(bi * k + p) * n + j];
        } } } }
    }
    Tensor::from_vec_f64(out, &shape)
}

pub(crate) fn matmul_general(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check("matmul", &[a, b], DType::F64)?;
    if a.ndim() == 0 || b.ndim() == 0 {
        return Err(Error::InvalidShape { op: "matmul", msg: "inputs must have at least one dimension".into() });
    }
    let (av, bv) = (a.ndim() == 1, b.ndim() == 1);
    let left = if av { a.reshape(&[1, a.shape()[0]])? } else { a.clone() };
    let right = if bv { b.reshape(&[b.shape()[0], 1])? } else { b.clone() };
    let (al, bl) = (left.ndim(), right.ndim());
    let (m, k, n) = (left.shape()[al - 2], left.shape()[al - 1], right.shape()[bl - 1]);
    if k != right.shape()[bl - 2] { return Err(Error::ShapeMismatch { op: "matmul", lhs: a.shape().to_vec(), rhs: b.shape().to_vec() }); }
    let batch_shape = broadcast_shapes("matmul", &left.shape()[..al - 2], &right.shape()[..bl - 2])?;
    let batch = checked_numel("matmul", &batch_shape)?;
    let mut a_shape = batch_shape.clone(); a_shape.extend([m, k]);
    let mut b_shape = batch_shape.clone(); b_shape.extend([k, n]);
    let mut shape = batch_shape; shape.extend([m, n]);
    let count = checked_numel("matmul", &shape)?;
    if count == 0 || k == 0 {
        if av { shape.remove(shape.len() - 2); }
        if bv { shape.pop(); }
        let (a_shape, b_shape) = (a.shape().to_vec(), b.shape().to_vec());
        return Ok(full_f64(&shape, 0.0)?.record_fn(vec![a.clone(), b.clone()], move |_| {
            vec![full_f64(&a_shape, 0.0).unwrap(), full_f64(&b_shape, 0.0).unwrap()]
        }));
    }
    for materialized in [&a_shape, &b_shape, &shape] {
        if checked_numel("matmul", materialized)? > isize::MAX as usize / 8 {
            return Err(Error::InvalidShape { op: "matmul", msg: "broadcast materialization exceeds allocation capacity".into() });
        }
    }
    let out = left.expand(&a_shape)?.reshape(&[batch, m, k])?.bmm(&right.expand(&b_shape)?.reshape(&[batch, k, n])?)?.reshape(&shape)?;
    let out = if av { out.squeeze(out.ndim() - 2)? } else { out };
    if bv { out.squeeze(out.ndim() - 1) } else { Ok(out) }
}

impl Tensor {
    /// CPU conversion preserving floating-point autograd. Integer and boolean
    /// destinations are nondifferentiable. The legacy to_dtype remains detached.
    pub fn to_dtype_autograd(&self, dtype: DType, copy: bool) -> Result<Tensor> {
        if dtype == self.dtype() && !copy { return Ok(self.clone()); }
        check("to_dtype_autograd", &[self], self.dtype())?;
        if category(self.dtype()).is_none() || category(dtype).is_none() {
            return Err(Error::Unsupported { op: "to_dtype_autograd", msg: "conversion supports float32, float64, int64 and bool".into() });
        }
        if dtype == DType::I64 && matches!(self.dtype(), DType::F32 | DType::F64) {
            if self.to_vec_f64().iter().any(|&x| !x.is_finite() || x < i64::MIN as f64 || x >= -(i64::MIN as f64)) {
                return Err(Error::Unsupported { op: "to_dtype_autograd", msg: "floating values outside the int64 range have no portable integer conversion".into() });
            }
        }
        let stride = if self.is_structural_zero() { default_strides(self.shape()) } else { copy_pointwise_layout(self) };
        let physical = pointwise_source(self, self.shape(), &stride)?;
        let raw = physical.to_dtype(dtype);
        let out = Tensor::from_parts(raw.0.storage.clone(), self.shape().to_vec(), stride, 0, Device::Cpu, false, None);
        if !matches!(dtype, DType::F32 | DType::F64) { return Ok(out); }
        let original = self.dtype();
        Ok(out.record_fn_forward(vec![self.clone()], crate::autograd::ForwardOp::Cast(dtype), move |g| vec![g.to_dtype_autograd(original, false).expect("validated cast adjoint")]))
    }

    pub fn dropout_f64(&self, p: f64, train: bool, seed: u64, offset: u64) -> Result<Tensor> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidShape { op: "dropout", msg: "probability must be finite and in [0, 1]".into() });
        }
        if self.dtype() != DType::F64 { return Err(Error::DtypeMismatch { op: "dropout", expected: DType::F64, got: self.dtype() }); }
        if !train || p == 0.0 || self.numel() == 0 { return Ok(self.clone()); }
        if p != 1.0 { crate::forward_ad::check_random_operation()?; }
        check("dropout", &[self], DType::F64)?;
        let philox = crate::philox::Philox::new(seed);
        let scale = if p == 1.0 { 0.0 } else { 1.0 / (1.0 - p) };
        let mask = Tensor::from_vec_f64((0..self.numel()).map(|i| {
            if (philox.uniform_at(offset, i as u64) as f64) < p { 0.0 } else { scale }
        }).collect(), self.shape())?;
        if crate::forward_ad::is_active() { return self.mul(&mask); }
        let out = binary("dropout", self, &mask, BinaryKind::Mul)?;
        Ok(out.record_fn(vec![self.clone()], move |g| vec![binary("dropout backward", g, &mask, BinaryKind::Mul).unwrap()]))
    }

    pub fn rand_f64(shape: &[usize], rng: &crate::Rng) -> Result<Tensor> {
        crate::forward_ad::check_random_operation()?;
        let mut values = filled("rand_f64", shape, 0.0f64)?;
        for value in &mut values { *value = rng.uniform_f64(); }
        Tensor::from_vec_f64(values, shape)
    }

    pub fn randn_f64(shape: &[usize], rng: &crate::Rng) -> Result<Tensor> {
        crate::forward_ad::check_random_operation()?;
        let mut values = filled("randn_f64", shape, 0.0f64)?;
        for value in &mut values { *value = rng.normal_f64(); }
        Tensor::from_vec_f64(values, shape)
    }

    /// Float64 scalar-power path, retaining the exponent's precision.
    pub fn powf64(&self, p: f64) -> Result<Tensor> {
        unary_grad_tensor(self, "powf64", move |x| {
            if p == 0.5 { x.sqrt() } else if p == -0.5 { x.sqrt().recip() } else if p == -1.0 { x.recip() } else if p == -2.0 { (x * x).recip() } else if p == 2.0 { x * x } else if p == 3.0 { (x * x) * x } else { x.powf(p) }
        }, false, move |g, x| {
            if p == 0.0 { full_f64(x.shape(), 0.0) }
            else { g.mul(&x.powf64(p - 1.0)?.mul(&full_f64(&[], p)?)?) }
        })
    }

    pub fn clamp_f64(&self, min: f64, max: f64) -> Result<Tensor> {
        unary_grad_tensor(self, "clamp_f64", move |x| if x.is_nan() || min.is_nan() || max.is_nan() { f64::NAN } else { let lower = if x < min { min } else { x }; if lower > max { max } else { lower } }, false,
            move |g, x| {
                // The numerical predicate is the first conditional operand,
                // so its materialized physical order precedes the cotangent.
                let mask = pointwise_binary_real("clamp_f64_mask", x, &full_f64(&[], min)?,
                    |_,_| unreachable!("float64 predicate"), move |x,min| if x >= min && x <= max { 1.0 } else { 0.0 })?;
                pointwise_binary_real_impl("clamp_f64_backward", &mask, &numeric_zero_operand(g)?, false,
                    |_,_| unreachable!("float64 conditional"), |mask,g| if mask != 0.0 { g } else { 0.0 })
            })
    }
}

pub(crate) fn softmax(input: &Tensor, dim: usize, logarithmic: bool) -> Result<Tensor> {
    let op = if logarithmic { "log_softmax" } else { "softmax" };
    check(op, &[input], DType::F64)?;
    if dim >= input.ndim().max(1) {
        return Err(Error::InvalidShape { op, msg: "axis out of range".into() });
    }
    let shape = input.shape().to_vec();
    let (outer, extent, inner) = if shape.is_empty() { (1, 1, 1) } else {
        (shape[..dim].iter().product::<usize>(), shape[dim], shape[dim + 1..].iter().product::<usize>())
    };
    let data = input.to_vec_f64();
    let mut output = filled(op, &shape, 0.0f64)?;
    if !output.is_empty() { for o in 0..outer { for i in 0..inner {
        let base = o * extent * inner + i;
        let maximum = (0..extent).map(|k| data[base + k * inner]).fold(f64::NEG_INFINITY, f64::max);
        let mut total = 0.0;
        for k in 0..extent {
            let index = base + k * inner;
            output[index] = (data[index] - maximum).exp();
            total += output[index];
        }
        let logarithm = total.ln();
        for k in 0..extent {
            let index = base + k * inner;
            output[index] = if logarithmic { (data[index] - maximum) - logarithm } else { output[index] / total };
        }
    } } }
    let result = Tensor::from_vec_f64(output.clone(), &shape)?;
    Ok(result.record_fn(vec![input.clone()], move |g| {
        let gradient = g.to_vec_f64();
        let mut dx = filled(op, &shape, 0.0f64).unwrap();
        if !dx.is_empty() { for o in 0..outer { for i in 0..inner {
            let base = o * extent * inner + i;
            let total: f64 = (0..extent).map(|k| {
                let index = base + k * inner;
                if logarithmic { gradient[index] } else { gradient[index] * output[index] }
            }).sum();
            for k in 0..extent {
                let index = base + k * inner;
                dx[index] = if logarithmic { gradient[index] - output[index].exp() * total }
                            else { output[index] * (gradient[index] - total) };
            }
        } } }
        vec![Tensor::from_vec_f64(dx, &shape).unwrap()]
    }).save_output_version())
}

fn dense_order(shape: &[usize], strides: &[usize], axes: &[usize]) -> bool {
    if shape.contains(&0) { return true; }
    let mut expected = 1usize;
    axes.iter().all(|&axis| {
        if shape[axis] == 1 { return true; }
        if strides[axis] != expected { return false; }
        expected *= shape[axis];
        true
    })
}

// Copy/cast preserve dense permutations, compacting gaps and overlaps with
// the single-input stride ordering. Empty copies retain their exact strides.
pub(crate) fn copy_pointwise_layout(input: &Tensor) -> Vec<usize> {
    let shape = input.shape();
    let mut sorted: Vec<_> = (0..shape.len()).collect();
    sorted.sort_by_key(|&axis| input.strides()[axis]);
    if dense_order(shape, input.strides(), &sorted) { return input.strides().to_vec(); }
    let mut axes: Vec<_> = (0..shape.len()).rev().collect();
    for index in 1..axes.len() {
        let mut current = index;
        for previous in (0..index).rev() {
            let (left, right) = (axes[previous], axes[current]);
            let (ls, rs) = (input.strides()[left], input.strides()[right]);
            if ls == 0 || rs == 0 { continue; }
            if ls < rs { break; }
            if ls > rs || shape[left] > shape[right] { axes.swap(previous, current); current = previous; }
        }
    }
    let mut stride = vec![0; shape.len()]; let mut expected = 1;
    for axis in axes { stride[axis] = expected; expected *= shape[axis].max(1); }
    stride
}

// TensorIterator's ordered per-input comparison makes the first concrete
// operand decisive; zero strides defer to the next operand. Promotion plans
// against the temporary copy's layout while values are read from the source.
pub(crate) fn binary_pointwise_layout(a: &Tensor, b: &Tensor, shape: &[usize], dtype: DType) -> Vec<usize> {
    binary_pointwise_layout_impl(a, b, shape, dtype, true)
}

fn binary_pointwise_layout_impl(a: &Tensor, b: &Tensor, shape: &[usize], dtype: DType, fast: bool) -> Vec<usize> {
    let strides = [a, b].map(|input| if input.dtype() == dtype { input.strides().to_vec() } else { copy_pointwise_layout(input) });
    iterator_pointwise_layout_impl(&[a, b], &strides, shape, fast)
}

// Conditions retain their own dtype and raw layout in a ternary iterator.
pub(crate) fn iterator_pointwise_layout(inputs: &[&Tensor], strides: &[Vec<usize>], shape: &[usize]) -> Vec<usize> {
    iterator_pointwise_layout_impl(inputs, strides, shape, true)
}

fn iterator_pointwise_layout_impl(inputs: &[&Tensor], strides: &[Vec<usize>], shape: &[usize], fast: bool) -> Vec<usize> {
    if fast && inputs.iter().all(|input| input.shape() == shape) {
        let axes: Vec<_> = (0..shape.len()).rev().collect();
        if strides.iter().all(|stride| dense_order(shape, stride, &axes)) { return default_strides(shape); }
        if shape.len() == 4 && strides.iter().all(|stride| dense_order(shape, stride, &[1, 3, 2, 0])) {
            let mut out = vec![0; 4]; let mut expected = 1;
            for axis in [1, 3, 2, 0] { out[axis] = expected; expected *= shape[axis].max(1); }
            return out;
        }
        let mut axes: Vec<_> = (0..shape.len()).collect();
        axes.sort_by_key(|&axis| strides[0][axis]);
        if strides.iter().all(|stride| *stride == strides[0] && dense_order(shape, stride, &axes)) { return strides[0].clone(); }
    }
    let expanded: Vec<Vec<usize>> = inputs.iter().zip(strides).map(|(input, stride)| {
        let mut expanded = vec![0; shape.len()]; let pad = shape.len() - input.ndim();
        for (axis, &size) in input.shape().iter().enumerate() {
            if size != 1 || shape[pad + axis] == 1 { expanded[pad + axis] = stride[axis]; }
        }
        expanded
    }).collect();
    let mut axes: Vec<_> = (0..shape.len()).rev().collect();
    for index in 1..axes.len() {
        let mut current = index;
        for previous in (0..index).rev() {
            let (left, right) = (axes[previous], axes[current]);
            let mut comparison = 0;
            for stride in &expanded {
                let (ls, rs) = (stride[left], stride[right]);
                if ls == 0 || rs == 0 { continue; }
                if ls < rs { comparison = -1; break; }
                if ls > rs || shape[left] > shape[right] { comparison = 1; break; }
            }
            if comparison < 0 { break; }
            if comparison > 0 { axes.swap(previous, current); current = previous; }
        }
    }
    if axes.iter().copied().eq((0..shape.len()).rev()) { return default_strides(shape); }
    let mut strides = vec![0; shape.len()]; let mut expected = 1;
    for axis in axes { strides[axis] = expected; expected *= shape[axis]; }
    strides
}

pub(crate) fn unary_pointwise_layout(input: &Tensor) -> Vec<usize> {
    if input.is_structural_zero() { default_strides(input.shape()) }
    else { binary_pointwise_layout(input, input, input.shape(), input.dtype()) }
}

pub(crate) fn pointwise_source(input: &Tensor, shape: &[usize], output_stride: &[usize]) -> Result<Tensor> {
    let source = input.broadcast_to(shape)?;
    let mut axes: Vec<_> = (0..shape.len()).collect();
    axes.sort_by_key(|&axis| output_stride[axis]);
    axes.reverse();
    Ok(Tensor::from_parts(source.0.storage.clone(), axes.iter().map(|&axis| shape[axis]).collect(),
        axes.iter().map(|&axis| source.strides()[axis]).collect(), source.storage_offset(), Device::Cpu, false, None))
}

pub(crate) fn pointwise_values_f32(input: &Tensor) -> Vec<f32> {
    match input.dtype() {
        DType::F32 => input.to_vec(), DType::F64 => input.to_vec_f64().into_iter().map(|x| x as f32).collect(),
        DType::I64 => input.to_vec_i64().into_iter().map(|x| x as f32).collect(),
        DType::Bool => input.to_vec_bool().into_iter().map(|x| u8::from(x) as f32).collect(), _ => unreachable!(),
    }
}
pub(crate) fn pointwise_values_f64(input: &Tensor) -> Vec<f64> {
    match input.dtype() {
        DType::F32 => input.to_vec().into_iter().map(f64::from).collect(), DType::F64 => input.to_vec_f64(),
        DType::I64 => input.to_vec_i64().into_iter().map(|x| x as f64).collect(),
        DType::Bool => input.to_vec_bool().into_iter().map(|x| u8::from(x) as f64).collect(), _ => unreachable!(),
    }
}

pub(crate) fn pointwise_binary_real(op: &'static str, a: &Tensor, b: &Tensor, f32_fn: impl Fn(f32, f32) -> f32, f64_fn: impl Fn(f64, f64) -> f64) -> Result<Tensor> {
    pointwise_binary_real_impl(op, a, b, true, f32_fn, f64_fn)
}

// A conditional with a scalar else-operand has the same two concrete stride
// comparisons, but cannot use the all-tensor contiguous/dense fast paths.
fn pointwise_binary_real_impl(op: &'static str, a: &Tensor, b: &Tensor, fast: bool, f32_fn: impl Fn(f32, f32) -> f32, f64_fn: impl Fn(f64, f64) -> f64) -> Result<Tensor> {
    if a.device() != b.device() { return Err(Error::DeviceMismatch { op, lhs: a.device(), rhs: b.device() }); }
    if a.device() != Device::Cpu { return Err(Error::Unsupported { op, msg: "typed pointwise host kernel requires CPU inputs".into() }); }
    let dtype = result_type(a, b)?;
    if !matches!(a.dtype(), DType::F32 | DType::F64) || !matches!(b.dtype(), DType::F32 | DType::F64) {
        return Err(Error::DtypeMismatch { op, expected: DType::F32, got: if !matches!(a.dtype(), DType::F32 | DType::F64) { a.dtype() } else { b.dtype() } });
    }
    if dtype == DType::F64 && crate::capture::is_recording() { return Err(Error::Unsupported { op, msg: "float64 capture is unsupported".into() }); }
    let shape = broadcast_shapes(op, a.shape(), b.shape())?;
    let count = check_shape(op, &shape)?;
    if count != 0 { check_shape(op, a.shape())?; check_shape(op, b.shape())?; }
    // Efficient zeros materialize numerically here, but contribute no
    // physical iteration preference to the ordinary kernel result.
    let stride = match (a.is_structural_zero(), b.is_structural_zero()) {
        (true, true) => default_strides(&shape),
        (true, false) => copy_pointwise_layout(&b.broadcast_to(&shape)?),
        (false, true) => copy_pointwise_layout(&a.broadcast_to(&shape)?),
        (false, false) => binary_pointwise_layout_impl(a, b, &shape, dtype, fast),
    };
    let (a, b) = (pointwise_source(a, &shape, &stride)?, pointwise_source(b, &shape, &stride)?);
    let raw = match dtype {
        DType::F32 => {
            let mut values = filled(op, &shape, 0.0f32)?;
            for ((value, x), y) in values.iter_mut().zip(pointwise_values_f32(&a)).zip(pointwise_values_f32(&b)) { *value = f32_fn(x, y); }
            Tensor::from_vec(values, &shape)?
        }
        DType::F64 => {
            let mut values = filled(op, &shape, 0.0f64)?;
            for ((value, x), y) in values.iter_mut().zip(pointwise_values_f64(&a)).zip(pointwise_values_f64(&b)) { *value = f64_fn(x, y); }
            Tensor::from_vec_f64(values, &shape)?
        }
        _ => unreachable!(),
    };
    Ok(Tensor::from_parts(raw.0.storage.clone(), shape, stride, 0, Device::Cpu, false, None))
}

#[cfg(test)]
mod pointwise_layout_tests {
    use super::*;
    #[test]
    fn structural_zero_add_sub_copy_layout_preserves_real_edges_and_immutable_markers() {
        for dtype in [DType::F32,DType::F64] {
            let source=Tensor::from_vec(vec![0.,-0.,1.,2.,-3.,4.],&[2,3]).unwrap().to_dtype(dtype);
            let x=source.transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
            let z=Tensor::structural_zero(x.shape(),dtype,Device::Cpu).unwrap();
            for (actual,negative) in [(x.add(&z).unwrap(),false),(z.add(&x).unwrap(),false),(x.sub(&z).unwrap(),false),(z.sub(&x).unwrap(),true)] {
                assert!(!actual.is_structural_zero());assert_eq!(actual.strides(),&[1,3]);
                assert_ne!(actual._storage_ptr(),x._storage_ptr());
                for (actual,wanted) in actual.to_vec_f64().into_iter().zip(x.to_vec_f64()) {assert_eq!(actual.to_bits(),(if negative {-wanted}else{wanted}).to_bits());}
                let gradient=actual.sum().grad_wrt(&[&x],false).unwrap().remove(0);
                assert_eq!(gradient.to_vec_f64(),vec![if negative {-1.}else{1.};6]);
            }
            let marked=x.mul(&z).unwrap();
            assert!(marked.is_structural_zero()&&marked.requires_grad());assert_eq!(marked.strides(),&[2,1]);
            assert!(marked.detach().zero_().is_err());
            let materialized=marked.clone_tensor(true).unwrap();assert!(!materialized.is_structural_zero());
            assert_eq!(materialized.to_vec_f64(),vec![0.;6]);
            let promoted=marked.mul(&Tensor::scalar(1.).to_dtype(if dtype==DType::F32{DType::F64}else{DType::F32}).expand(marked.shape()).unwrap()).unwrap();
            assert!(promoted.is_structural_zero()&&promoted.requires_grad());
        }
        let base=Tensor::ones(&[2,16]).requires_grad_(true).unwrap();
        let gapped=base.narrow(1,0,1).unwrap();
        let zero=Tensor::structural_zero(&[2,0],DType::F64,Device::Cpu).unwrap();
        let promoted=gapped.add(&zero).unwrap();
        assert_eq!(promoted.strides(),&[16,0]);assert!(!promoted.is_structural_zero());
        assert!(promoted.requires_grad());promoted.sum().backward();
        assert_eq!(base.grad().unwrap().to_vec(),vec![0.;32]);
    }
}
