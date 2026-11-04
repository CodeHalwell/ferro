//! CPU binary losses with staged dtype promotion and one native first-order Op.
use crate::{DType, Error, Result, Tensor};

const OP: &str = "binary_loss";
fn invalid(message: impl Into<String>) -> Error { Error::Unsupported { op: OP, msg: message.into() } }
fn cast(value: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 { value as f32 as f64 } else { value }
}
fn promote(float: DType, scalar: bool, other: DType, other_scalar: bool) -> DType {
    if matches!(other, DType::I64 | DType::Bool) { return float; }
    if scalar && !other_scalar { return other; }
    if other_scalar && !scalar { return float; }
    if float == DType::F64 || other == DType::F64 { DType::F64 } else { DType::F32 }
}
fn add(a: f64, b: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 { ((a as f32) + (b as f32)) as f64 } else { a + b }
}
fn sub(a: f64, b: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 { ((a as f32) - (b as f32)) as f64 } else { a - b }
}
fn mul(a: f64, b: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 { ((a as f32) * (b as f32)) as f64 } else { a * b }
}
fn div(a: f64, b: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 { ((a as f32) / (b as f32)) as f64 } else { a / b }
}
fn log_sigmoid(value: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 {
        let x = value as f32;
        -((-x).max(0.) + (-x.abs()).exp().ln_1p()) as f64
    } else { -((-value).max(0.) + (-value.abs()).exp().ln_1p()) }
}
fn sigmoid(value: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 { (1. / (1. + (-(value as f32)).exp())) as f64 }
    else { 1. / (1. + (-value).exp()) }
}
fn logs(value: f64, dtype: DType) -> (f64, f64) {
    if dtype == DType::F32 { let x = value as f32; (x.ln() as f64, (-x).ln_1p() as f64) }
    else { (value.ln(), (-value).ln_1p()) }
}

enum Values { F32(Vec<f32>), F64(Vec<f64>), I64(Vec<i64>), Bool(Vec<bool>) }
impl Values {
    fn read(tensor: &Tensor, shape: &[usize]) -> Result<Self> {
        let value = tensor.broadcast_to(shape)?;
        Ok(match value.dtype() {
            DType::F32 => Self::F32(value.to_vec()), DType::F64 => Self::F64(value.to_vec_f64()),
            DType::I64 => Self::I64(value.to_vec_i64()), DType::Bool => Self::Bool(value.to_vec_bool()),
            _ => return Err(invalid("weights require float32, float64, int64 or bool")),
        })
    }
    fn at(&self, index: usize, dtype: DType) -> f64 {
        match self {
            Self::F32(v) => cast(v[index] as f64, dtype), Self::F64(v) => cast(v[index], dtype),
            Self::I64(v) => if dtype == DType::F32 { v[index] as f32 as f64 } else { v[index] as f64 },
            Self::Bool(v) => if v[index] { 1. } else { 0. },
        }
    }
    fn minus_one(&self, index: usize, dtype: DType) -> f64 {
        match self {
            Self::F32(v) => cast((v[index] - 1.) as f64, dtype), Self::F64(v) => cast(v[index] - 1., dtype),
            Self::I64(v) => {
                let value = v[index].wrapping_sub(1);
                if dtype == DType::F32 { value as f32 as f64 } else { value as f64 }
            }
            Self::Bool(_) => unreachable!(),
        }
    }
}
fn tensor(values: Vec<f64>, shape: &[usize], dtype: DType) -> Result<Tensor> {
    if dtype == DType::F64 { Tensor::from_vec_f64(values, shape) }
    else { Tensor::from_vec(values.into_iter().map(|x| x as f32).collect(), shape) }
}

fn binary_loss(input: &Tensor, target: &Tensor, positive: Option<&Tensor>, weight: Option<&Tensor>, logits: bool, reduction: &str) -> Result<Tensor> {
    let reduced = match reduction { "none" => 0, "sum" => 1, "mean" => 2, _ => return Err(invalid("reduction must be none, sum or mean")) };
    let (xd, td) = (input.dtype(), target.dtype());
    if !matches!(xd, DType::F32 | DType::F64) || !matches!(td, DType::F32 | DType::F64) {
        return Err(invalid("binary loss inputs and targets require float32 or float64"));
    }
    if !logits && xd != td { return Err(Error::DtypeMismatch { op: OP, expected: xd, got: td }); }
    if input.shape() != target.shape() {
        return Err(Error::ShapeMismatch { op: OP, lhs: input.shape().to_vec(), rhs: target.shape().to_vec() });
    }
    if !logits && positive.is_some() { return Err(invalid("pos_weight requires logits")); }
    let mut inputs = vec![input, target];
    if let Some(value) = positive { inputs.push(value); }
    if let Some(value) = weight { inputs.push(value); }
    for value in &inputs { crate::typed_math::check(OP, &[*value], value.dtype())?; }
    for (value, is_positive) in [(positive, true), (weight, false)] {
        if let Some(value) = value {
            if !matches!(value.dtype(), DType::F32 | DType::F64 | DType::I64 | DType::Bool) || (is_positive && value.dtype() == DType::Bool) {
                return Err(invalid("weights require float32/float64/int64/bool; bool pos_weight is unsupported"));
            }
            if value.requires_grad() && crate::autograd::is_grad_enabled() { return Err(invalid("weight and pos_weight are not differentiable")); }
            if crate::shape::broadcast_shapes(OP, input.shape(), value.shape())? != input.shape() {
                return Err(Error::ShapeMismatch { op: OP, lhs: input.shape().to_vec(), rhs: value.shape().to_vec() });
            }
        }
    }
    let scalar = input.ndim() == 0;
    let output_dtype = if logits { td } else { xd };
    let ptype = positive.map(|p| promote(td, scalar, p.dtype(), p.ndim() == 0));
    let p_log_type = positive.map(|p| promote(xd, scalar, p.dtype(), p.ndim() == 0));
    let wmeta = weight.map(|w| (w.dtype(), if logits { w.ndim() == 0 } else { scalar }));
    let x = Values::read(input, input.shape())?;
    let t = Values::read(target, target.shape())?;
    let p = positive.map(|v| Values::read(v, input.shape())).transpose()?;
    let w = weight.map(|v| Values::read(v, input.shape())).transpose()?;
    let count = input.numel();
    let mut output = crate::typed_math::filled(OP, input.shape(), 0.0f64)?;
    let common = promote(xd, scalar, td, scalar);
    for (i, out) in output.iter_mut().enumerate() {
        let (xv, tv) = (x.at(i, xd), t.at(i, td));
        if logits {
            let mut ls = log_sigmoid(xv, xd);
            if let Some(values) = &p {
                let pd = ptype.unwrap();
                let factor = add(mul(values.minus_one(i, pd), tv, pd), 1., pd);
                ls = cast(mul(ls, factor, promote(xd, scalar, pd, scalar)), xd);
            }
            let negative = sub(1., tv, td);
            *out = cast(sub(cast(mul(negative, xv, common), td), ls, common), td);
        } else {
            if !(xv >= 0. && xv <= 1. && tv >= 0. && tv <= 1.) { return Err(invalid("binary cross entropy inputs and targets must be between 0 and 1")); }
            let (lp, lm) = logs(xv, xd);
            *out = -add(mul(tv, lp.max(-100.), xd), mul(sub(1., tv, xd), lm.max(-100.), xd), xd);
        }
        if let Some(values) = &w {
            let (wd, ws) = wmeta.unwrap();
            let arithmetic = promote(output_dtype, scalar, wd, ws);
            *out = cast(mul(*out, values.at(i, arithmetic), arithmetic), output_dtype);
        }
    }
    let output = tensor(output, input.shape(), output_dtype)?;
    let output = match reduced { 0 => output, 1 => output.sum(), _ => output.mean() };
    let shape = input.shape().to_vec();
    let parents = inputs.into_iter().cloned().collect();
    Ok(output.record_fn_optional(parents, move |gradient| {
        let g = Values::read(gradient, gradient.shape()).unwrap();
        let gscalar = reduced != 0 || scalar;
        let mut dx = crate::typed_math::filled(OP, &shape, 0.0f64).unwrap();
        let mut dt = crate::typed_math::filled(OP, &shape, 0.0f64).unwrap();
        for i in 0..count {
            let (xv, tv) = (x.at(i, xd), t.at(i, td));
            let seed = g.at(if reduced == 0 { i } else { 0 }, output_dtype);
            let (mut gx, mut gt, gxd, gtd);
            if logits {
                if let Some(values) = &p {
                    let pd = ptype.unwrap();
                    let pt = mul(values.at(i, pd), tv, pd);
                    gx = sub(cast(mul(sub(add(pt, 1., pd), tv, pd), sigmoid(xv, xd), promote(pd, scalar, xd, scalar)), pd), pt, pd);
                    gxd = pd;
                    let lp = p_log_type.unwrap();
                    gt = cast(sub(log_sigmoid(-xv, xd), mul(log_sigmoid(xv, xd), values.at(i, lp), lp), promote(xd, scalar, lp, scalar)), xd);
                    gt = cast(mul(gt, seed, promote(xd, scalar, output_dtype, gscalar)), xd);
                    gtd = xd;
                } else {
                    gx = cast(sub(sigmoid(xv, xd), tv, common), xd);
                    gxd = xd;
                    gtd = promote(xd, scalar, output_dtype, gscalar);
                    gt = mul(-xv, seed, gtd);
                }
                gx = cast(mul(gx, seed, promote(gxd, scalar, output_dtype, gscalar)), gxd);
            } else {
                gx = div(mul(seed, sub(xv, tv, xd), xd), mul(xv, sub(1., xv, xd), xd).max(1e-12f32 as f64), xd);
                let ratio = div(xv, sub(1., xv, xd), xd);
                let log_ratio = if xd == DType::F32 { (ratio as f32).ln() as f64 } else { ratio.ln() };
                gt = mul(-log_ratio, seed, xd);
                gxd = xd; gtd = td;
            }
            if let Some(values) = &w {
                let (wd, ws) = wmeta.unwrap();
                let wx = promote(gxd, scalar, wd, ws);
                let wt = promote(gtd, scalar, wd, ws);
                gx = cast(mul(gx, values.at(i, wx), wx), gxd);
                gt = cast(mul(gt, values.at(i, wt), wt), gtd);
            }
            if reduced == 2 {
                gx = div(gx, if gxd == DType::F32 { count as f32 as f64 } else { count as f64 }, gxd);
                gt = div(gt, if gtd == DType::F32 { count as f32 as f64 } else { count as f64 }, gtd);
            }
            dx[i] = cast(gx, xd); dt[i] = cast(gt, td);
        }
        let mut result = vec![Some(tensor(dx, &shape, xd).unwrap()), Some(tensor(dt, &shape, td).unwrap())];
        if p.is_some() { result.push(None); }
        if w.is_some() { result.push(None); }
        result
    }))
}

impl Tensor {
    pub fn binary_cross_entropy_unreduced(&self, target: &Tensor) -> Result<Tensor> {
        binary_loss(self, target, None, None, false, "none")
    }
    pub fn bce_with_logits_unreduced(&self, target: &Tensor, pos_weight: Option<&Tensor>) -> Result<Tensor> {
        binary_loss(self, target, pos_weight, None, true, "none")
    }
    pub fn binary_loss(&self, target: &Tensor, pos_weight: Option<&Tensor>, weight: Option<&Tensor>, logits: bool, reduction: &str) -> Result<Tensor> {
        binary_loss(self, target, pos_weight, weight, logits, reduction)
    }
}
