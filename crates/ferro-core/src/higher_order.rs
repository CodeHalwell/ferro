//! Graph-building VJPs on the existing record_fn Op graph, never a second tape.
use super::{ForwardOp, Op};
use crate::dispatch::{BinaryKind, OpTag, UnaryKind};
use crate::{Error, Result, Tensor};

pub(super) fn reduce_to(g: &Tensor, shape: &[usize]) -> Result<Tensor> {
    if g.numel() == 0 { return crate::tensor::sum_empty_to_shape(g, shape); }
    let mut v = g.clone();
    while v.shape().len() > shape.len() { v = v.sum_dim(0, false)?; }
    for (d, &size) in shape.iter().enumerate() {
        if size == 1 && v.shape()[d] != 1 { v = v.sum_dim(d, true)?; }
    }
    if v.shape() == shape { Ok(v) } else { v.reshape(shape) }
}

impl Op {
    pub(super) fn vjp_uses_output(&self) -> bool {
        matches!(self.tag, Some(OpTag::Unary(UnaryKind::Exp | UnaryKind::Relu | UnaryKind::Sqrt | UnaryKind::Tanh | UnaryKind::Sigmoid)))
            || matches!(&self.forward, Some(ForwardOp::Softmax(_) | ForwardOp::Rsqrt | ForwardOp::CumProd { .. } | ForwardOp::Prod { .. }))
    }

    pub(super) fn graph_backward(&self, g: &Tensor, inputs: &[Tensor], saved_output: Option<&Tensor>) -> Result<Vec<Tensor>> {
        let a = &inputs[0];
        let one = || Tensor::full_on(a.shape(), 1.0, a.device());
        let grads = match self.tag {
            Some(OpTag::Binary(kind)) => {
                let b = &inputs[1];
                let (ga, gb) = match kind {
                    BinaryKind::Add => (g.clone(), g.clone()),
                    BinaryKind::Sub => (g.clone(), g.neg()),
                    BinaryKind::Mul => (g.mul(b)?, g.mul(a)?),
                    BinaryKind::Div => (g.div(b)?, g.neg().mul(&a.div(b)?.div(b)?)?),
                    BinaryKind::ReluBackward => (crate::ops::relu_backward(g, b)?, Tensor::full_on(b.shape(), 0.0, b.device())?),
                };
                vec![reduce_to(&ga, a.shape())?, reduce_to(&gb, b.shape())?]
            }
            Some(OpTag::Unary(UnaryKind::Neg)) => vec![g.neg()],
            Some(OpTag::Unary(UnaryKind::Exp)) => vec![g.mul(saved_output.ok_or_else(|| Error::Unsupported { op: "exp backward", msg: "recorded output is unavailable".into() })?)?],
            Some(OpTag::Unary(UnaryKind::Log)) => vec![g.div(a)?],
            Some(OpTag::Unary(UnaryKind::Powf(power))) => {
                if power == 0. { vec![Tensor::full_on(a.shape(), 0., a.device())?] }
                else { vec![g.mul(&a.powf(power - 1.).mul(&Tensor::full_on(&[], power, a.device())?)?)?] }
            }
            Some(OpTag::Unary(UnaryKind::Relu)) => vec![crate::ops::relu_backward(g, &saved_output.cloned().unwrap_or_else(|| a.relu()))?],
            Some(OpTag::Unary(UnaryKind::Abs)) => vec![g.mul(&a.sign()?)?],
            Some(OpTag::Unary(UnaryKind::Sign)) => vec![Tensor::full_on(a.shape(), 0.0, a.device())?],
            Some(OpTag::Unary(UnaryKind::Clamp { min, max })) => {
                let lower = Tensor::full_on(&[], min, a.device())?;
                let upper = Tensor::full_on(&[], max, a.device())?;
                let mask = a.ge_tensor(&lower)?.logical_and(&a.le_tensor(&upper)?)?;
                vec![Tensor::where_cond(&mask, g, &Tensor::full_on(&[], 0., g.device())?)?]
            }
            Some(OpTag::Unary(UnaryKind::Silu)) => {
                let s = a.sigmoid();
                let derivative = s.add(&a.mul(&s)?.mul(&one()?.sub(&s)?)?)?;
                vec![g.mul(&derivative)?]
            }
            Some(OpTag::Unary(UnaryKind::Sqrt)) => {
                let y = saved_output.ok_or_else(|| Error::Unsupported { op: "sqrt backward", msg: "recorded output is unavailable".into() })?;
                vec![g.div(&y.mul(&Tensor::full_on(&[], 2.0, a.device())?)?)?]
            }
            Some(OpTag::Unary(UnaryKind::Tanh | UnaryKind::Sigmoid)) => {
                let y = saved_output.ok_or_else(|| Error::Unsupported { op: "activation backward", msg: "recorded output is unavailable".into() })?;
                let kind = if matches!(self.tag, Some(OpTag::Unary(UnaryKind::Tanh))) { crate::ops_ext::activation_backward::Kind::Tanh } else { crate::ops_ext::activation_backward::Kind::Sigmoid };
                vec![crate::ops_ext::activation_backward::backward(kind, g, y)?]
            }
            _ => match &self.forward {
                Some(ForwardOp::SigmoidBackward | ForwardOp::TanhBackward) => {
                    let kind = if matches!(self.forward, Some(ForwardOp::TanhBackward)) { crate::ops_ext::activation_backward::Kind::Tanh } else { crate::ops_ext::activation_backward::Kind::Sigmoid };
                    crate::ops_ext::activation_backward::vjp(kind, g, &inputs[0], &inputs[1])?
                },
                Some(ForwardOp::IndexedGather { .. } | ForwardOp::IndexedScatter { .. } | ForwardOp::CumExtrema { .. }) => return Err(Error::Unsupported { op: "indexed backward", msg: "indexed selection uses its selected optional VJP".into() }),
                Some(ForwardOp::Sum) => vec![g.mul(&one()?)?],
                Some(ForwardOp::CumSum { dim, reverse, .. }) => vec![crate::ops_ext::cumsum::selected_vjp(g, a.shape(), a.dtype(), *dim, *reverse)?],
                Some(ForwardOp::Prod { dim, keepdim, .. }) => vec![crate::ops_ext::prod::graph_vjp(a, saved_output, g, *dim, *keepdim, super::functional_vjp())?],
                Some(ForwardOp::CumProd { dim, .. }) => {
                    let output = saved_output.ok_or_else(|| Error::Unsupported { op: "cumprod backward", msg: "recorded output is unavailable".into() })?;
                    vec![crate::ops_ext::cumprod::graph_vjp(a, output, g, *dim)?]
                },
                Some(ForwardOp::Square) => vec![g.mul(&a.powf(1.).mul(&Tensor::full_on(&[], 2., a.device())?)?)?],
                Some(ForwardOp::ScalarPower(power)) => vec![crate::ops_ext::powf::scalar_power_backward(g, a, *power)?],
                Some(ForwardOp::RoundedDiv(_) | ForwardOp::FloorDivide) => return Err(Error::Unsupported { op: "rounded division backward", msg: "rounded division uses its selected optional VJP".into() }),
                Some(ForwardOp::Rounding(_)) => vec![Tensor::full_on(a.shape(), 0.0, a.device())?],
                Some(ForwardOp::Frac) => vec![g.clone()],
                Some(ForwardOp::Rsqrt) => {
                    let y = saved_output.ok_or_else(|| Error::Unsupported { op: "rsqrt backward", msg: "recorded output is unavailable".into() })?;
                    vec![g.mul(&Tensor::full_on(&[], -0.5, a.device())?)?.mul(&y.mul(y)?.mul(y)?)?]
                },
                Some(ForwardOp::AddAlpha { alpha, subtract }) => {
                    let signed = if *subtract { g.neg() } else { g.clone() };
                    vec![reduce_to(g, a.shape())?.to_dtype_autograd(a.dtype(), false)?, reduce_to(&alpha.scale(&signed)?, inputs[1].shape())?.to_dtype_autograd(inputs[1].dtype(), false)?]
                },
                Some(ForwardOp::Mean) => vec![g.mul(&Tensor::full_on(a.shape(), 1.0 / a.numel() as f32, a.device())?)?],
                Some(ForwardOp::MatMul) => vec![g.matmul(&inputs[1].transpose(0, 1)?)?, a.transpose(0, 1)?.matmul(g)?],
                Some(ForwardOp::Bmm) => vec![g.bmm(&inputs[1].transpose(1, 2)?)?, a.transpose(1, 2)?.bmm(g)?],
                Some(ForwardOp::Softmax(dim)) => {
                    let y = if let Some(value) = saved_output { value.clone() } else { a.softmax(*dim)? };
                    let dot = g.mul(&y)?.sum_dim(*dim, true)?;
                    vec![y.mul(&g.sub(&dot)?)?]
                }
                Some(ForwardOp::Reshape(_) | ForwardOp::View(_)) => vec![g.reshape(a.shape())?],
                Some(ForwardOp::Squeeze(dim)) => vec![g.unsqueeze(*dim)?],
                Some(ForwardOp::Unsqueeze(dim)) => vec![g.squeeze(*dim)?],
                Some(ForwardOp::CopyDualLayout { .. }) => vec![g.to_dtype_autograd(a.dtype(),false)?],
                Some(ForwardOp::Alias | ForwardOp::Clone { .. }) => vec![g.clone()],
                Some(ForwardOp::Transpose(d0, d1)) => vec![g.transpose(*d0, *d1)?],
                Some(ForwardOp::Permute(dims)) => {
                    let mut inverse = vec![0; dims.len()];
                    for (index, &dim) in dims.iter().enumerate() { inverse[dim] = index; }
                    vec![g.permute(&inverse)?]
                }
                Some(ForwardOp::Expand(_)) => vec![reduce_to(g, a.shape())?],
                Some(ForwardOp::Cast(_)) => vec![g.to_dtype_autograd(a.dtype(), false)?],
                Some(ForwardOp::Sin) => vec![g.mul(&a.cos()?)?],
                Some(ForwardOp::Cos) => vec![g.neg().mul(&a.sin()?)?],
                Some(ForwardOp::Cat(dim)) => {
                    let mut offset = 0;
                    let mut parts = Vec::with_capacity(inputs.len());
                    for input in inputs {
                        parts.push(g.narrow(*dim, offset, input.shape()[*dim])?);
                        offset += input.shape()[*dim];
                    }
                    parts
                }
                Some(ForwardOp::Slice { dim, start, end, step }) => vec![g.slice_adjoint(a.shape(), *dim, *start, *end, *step)?],
                Some(ForwardOp::SliceAdjoint { dim, start, end, step, .. }) => vec![g.slice(*dim, *start, *end, *step)?],
                Some(ForwardOp::SumDim(dim, keep)) => {
                    let mut shape = a.shape().to_vec();
                    shape[*dim] = 1;
                    let expanded = if *keep { g.clone() } else { g.reshape(&shape)? };
                    vec![expanded.expand(a.shape())?]
                }
                _ => return Err(Error::Unsupported { op: "create_graph", msg: format!("no differentiable VJP registered for tag {:?}, forward {:?}", self.tag, self.forward) }),
            },
        };
        Ok(if let Some(output) = saved_output {
            grads.into_iter().map(|gradient| gradient.save_tensor_versions(&[output])).collect()
        } else { grads })
    }
}
