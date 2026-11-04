//! Local JVPs on the same Op that owns the ordinary backward rule.
use super::{ForwardOp, Op};
use crate::dispatch::{BinaryKind, OpTag, UnaryKind};
use crate::{Device, DType, Error, Result, Tensor};

fn constant(shape: &[usize], value: f32, dtype: DType) -> Result<Tensor> {
    if dtype == DType::F64 { crate::typed_math::full_f64(shape, value as f64) }
    else { Tensor::full_on(shape, value, Device::Cpu) }
}

fn expanded(tangent: &Tensor, shape: &[usize]) -> Result<Tensor> {
    if tangent.shape() == shape { Ok(tangent.clone()) } else { tangent.expand(shape) }
}

impl Op {
    pub(crate) fn jvp(&self, output: &Tensor, tangents: &[Option<Tensor>]) -> Result<Tensor> {
        let inputs = self.inputs();
        let a = &inputs[0];
        let tangent = |index: usize| -> Result<Tensor> {
            match &tangents[index] {
                Some(value) => Ok(value.clone()),
                None => Tensor::structural_zero(inputs[index].shape(), inputs[index].dtype(), inputs[index].device()),
            }
        };
        let g = || tangent(0);
        let result = match self.tag {
            Some(OpTag::Binary(kind)) => {
                let b = &inputs[1];
                match kind {
                    BinaryKind::Add => match (&tangents[0], &tangents[1]) {
                        (Some(left), Some(right)) => left.add(right),
                        (Some(value), None) | (None, Some(value)) => expanded(value, output.shape()),
                        _ => unreachable!("JVP requires an input tangent"),
                    },
                    BinaryKind::Sub => match (&tangents[0], &tangents[1]) {
                        (Some(left), Some(right)) => left.sub(right),
                        (Some(value), None) => expanded(value, output.shape()),
                        (None, Some(value)) => expanded(&value.neg(), output.shape()),
                        _ => unreachable!("JVP requires an input tangent"),
                    },
                    BinaryKind::Mul => g()?.mul(b)?.add(&tangent(1)?.mul(a)?),
                    BinaryKind::Div => g()?.sub(&tangent(1)?.mul(&a.div(b)?)?)?.div(b),
                    BinaryKind::ReluBackward => crate::ops::relu_backward(&g()?, b),
                }
            }
            Some(OpTag::Unary(UnaryKind::Neg)) => Ok(g()?.neg()),
            Some(OpTag::Unary(UnaryKind::Exp)) => g()?.mul(output),
            Some(OpTag::Unary(UnaryKind::Log)) => g()?.div(a),
            Some(OpTag::Unary(UnaryKind::Powf(power))) => {
                if power == 0. { constant(output.shape(), 0., output.dtype()) }
                else { g()?.mul(&a.powf(power - 1.).mul(&constant(&[], power, a.dtype())?)?) }
            }
            Some(OpTag::Unary(UnaryKind::Relu)) => crate::ops::relu_backward(&g()?, output),
            Some(OpTag::Unary(UnaryKind::Abs)) => g()?.mul(&a.sign()?),
            Some(OpTag::Unary(UnaryKind::Sign | UnaryKind::Gtz)) => constant(output.shape(), 0., g()?.dtype()),
            Some(OpTag::Unary(UnaryKind::Clamp { min, max })) => {
                let mask = a.ge_tensor(&constant(&[], min, a.dtype())?)?.logical_and(&a.le_tensor(&constant(&[], max, a.dtype())?)?)?;
                let gradient = g()?;
                Tensor::where_cond(&mask, &gradient, &constant(&[], 0., gradient.dtype())?)
            }
            Some(OpTag::Unary(UnaryKind::Sqrt)) => g()?.div(&output.mul(&constant(&[], 2., output.dtype())?)?),
            Some(OpTag::Unary(UnaryKind::Tanh)) => crate::ops_ext::activation_backward::backward(crate::ops_ext::activation_backward::Kind::Tanh, &g()?, output),
            Some(OpTag::Unary(UnaryKind::Sigmoid)) => crate::ops_ext::activation_backward::backward(crate::ops_ext::activation_backward::Kind::Sigmoid, &g()?, output),
            Some(OpTag::Unary(UnaryKind::Silu)) => {
                let s = a.sigmoid();
                g()?.mul(&s.add(&a.mul(&s)?.mul(&constant(&[], 1., a.dtype())?.sub(&s)?)?)?)
            }
            _ => match &self.forward {
                Some(ForwardOp::SigmoidBackward | ForwardOp::TanhBackward) => {
                    let kind = if matches!(self.forward, Some(ForwardOp::TanhBackward)) { crate::ops_ext::activation_backward::Kind::Tanh } else { crate::ops_ext::activation_backward::Kind::Sigmoid };
                    crate::ops_ext::activation_backward::jvp(kind, &inputs[0], &inputs[1], &tangent(0)?, &tangent(1)?)
                },
                Some(ForwardOp::CumExtrema { dim, .. }) => g()?.indexed_gather(&inputs[1], crate::ops_ext::indexed::IndexMap::Axis(*dim), true),
                Some(ForwardOp::Prelu) => crate::ops_ext::prelu::jvp(a, &inputs[1], tangents),
                Some(ForwardOp::WhereConditional) => crate::ops_ext::where_cond::jvp(inputs, output, tangents),
                Some(ForwardOp::IndexedGather { map, kind, save_input }) => g()?.indexed_gather_kind(&inputs[1], *map, *kind, *save_input),
                Some(ForwardOp::IndexedScatter { map, kind, shape, save_input, strides }) => Tensor::indexed_scatter_saved(&g()?, &inputs[1], shape, *map, *kind, *save_input, strides.as_deref()),
                Some(ForwardOp::Sum) => Ok(g()?.sum()),
                Some(ForwardOp::CumSum { dim, dtype, reverse }) => g()?.cumulative_sum(*dim, *dtype, *reverse),
                Some(ForwardOp::Prod { dim, keepdim, .. }) => crate::ops_ext::prod::jvp(a, output, &g()?, *dim, *keepdim),
                Some(ForwardOp::CumProd { dim, .. }) => crate::ops_ext::cumprod::jvp(a, output, &g()?, *dim),
                Some(ForwardOp::Square) => g()?.mul(&a.mul(&constant(&[], 2., a.dtype())?)?),
                Some(ForwardOp::ScalarPower(power)) => crate::ops_ext::powf::scalar_power_backward(&g()?, a, *power),
                Some(ForwardOp::RoundedDiv(_)) => crate::ops_ext::div_rounding::zero_like(output),
                Some(ForwardOp::FloorDivide) => Err(Error::Unsupported { op: "floor_divide forward_ad", msg: "forward AD for floor_divide is not implemented".into() }),
                Some(ForwardOp::Rounding(_)) => constant(output.shape(), 0.0, g()?.dtype()),
                Some(ForwardOp::Frac) => g(),
                Some(ForwardOp::Rsqrt) => g()?.mul(&constant(&[], -0.5, g()?.dtype())?)?.mul(&output.mul(output)?.mul(output)?),
                Some(ForwardOp::AddAlpha { alpha, subtract }) => {
                    alpha.validate_ad()?;
                    let prepared = |index: usize| -> Result<Option<Tensor>> {
                        tangents[index].as_ref().map(|value| {
                            // Keep multiplication before broadcasting, as in
                            // Torch, except when empty output avoids huge reads.
                            let value = if output.numel() == 0 { expanded(value, output.shape())? } else { value.clone() };
                            if inputs[index].dtype() == output.dtype() { Ok(value) }
                            else { value.to_dtype_autograd(output.dtype(), false) }
                        }).transpose()
                    };
                    match (prepared(0)?, prepared(1)?) {
                        (Some(left), Some(right)) => {
                            let scaled = alpha.scale(&right)?;
                            if *subtract { left.sub(&scaled) } else { left.add(&scaled) }
                        },
                        (Some(value), None) => expanded(&value, output.shape()),
                        (None, Some(value)) => { let scaled = alpha.scale(&value)?; expanded(&if *subtract { scaled.neg() } else { scaled }, output.shape()) },
                        _ => unreachable!("JVP requires an input tangent"),
                    }
                },
                Some(ForwardOp::Mean) => Ok(g()?.mean()),
                Some(ForwardOp::SumDim(dim, keep)) => g()?.sum_dim(*dim, *keep),
                Some(ForwardOp::MatMul) => g()?.matmul(&inputs[1])?.add(&a.matmul(&tangent(1)?)?),
                Some(ForwardOp::Bmm) => g()?.bmm(&inputs[1])?.add(&a.bmm(&tangent(1)?)?),
                Some(ForwardOp::Reshape(shape) | ForwardOp::View(shape)) => g()?.reshape(shape),
                Some(ForwardOp::Squeeze(dim)) => g()?.squeeze(*dim),
                Some(ForwardOp::Unsqueeze(dim)) => g()?.unsqueeze(*dim),
                Some(ForwardOp::Transpose(left, right)) => g()?.transpose(*left, *right),
                Some(ForwardOp::Permute(dims)) => g()?.permute(dims),
                Some(ForwardOp::Expand(shape)) => g()?.expand(shape),
                Some(ForwardOp::Cast(dtype)) => g()?.to_dtype_autograd(*dtype, true),
                Some(ForwardOp::Sin) => g()?.mul(&a.cos()?),
                Some(ForwardOp::Cos) => g()?.neg().mul(&a.sin()?),
                Some(ForwardOp::CopyDualLayout { strides, offset, storage_len }) => g()?.copy_dual_layout(strides,*offset,*storage_len),
                Some(ForwardOp::Clone { preserve_layout }) => g()?.clone_tensor(*preserve_layout),
                Some(ForwardOp::Cat(dim)) => {
                    let dtype = if tangents.iter().flatten().any(|value| value.dtype() == DType::F64) { DType::F64 } else { output.dtype() };
                    let values = (0..inputs.len()).map(|index| tangent(index)?.to_dtype_autograd(dtype, false)).collect::<Result<Vec<_>>>()?;
                    Tensor::cat(&values, *dim)
                }
                Some(ForwardOp::Slice { dim, start, end, step }) => g()?.slice(*dim, *start, *end, *step),
                Some(ForwardOp::SliceAdjoint { shape, dim, start, end, step }) => g()?.slice_adjoint(shape, *dim, *start, *end, *step),
                Some(ForwardOp::Softmax(dim)) => {
                    let g = g()?;
                    output.mul(&g.sub(&g.mul(output)?.sum_dim(*dim, true)?)?)
                }
                Some(ForwardOp::LayerNorm { weight, bias, eps }) => {
                    let dim = a.ndim() - 1;
                    let width = constant(&[], a.shape()[dim] as f32, a.dtype())?;
                    let mean = a.sum_dim(dim, true)?.div(&width)?;
                    let centered = a.sub(&mean)?;
                    let scale = centered.mul(&centered)?.sum_dim(dim, true)?.div(&width)?.add(&constant(&[], *eps, a.dtype())?)?.sqrt();
                    let normalized = centered.div(&scale)?;
                    let da = g()?;
                    let dmean = da.sum_dim(dim, true)?.div(&width)?;
                    let correction = centered.mul(&da)?.sum_dim(dim, true)?.div(&width)?.div(&scale.mul(&scale)?)?;
                    let mut result = da.sub(&dmean)?.sub(&centered.mul(&correction)?)?.div(&scale)?;
                    if *weight { result = result.mul(&inputs[1])?.add(&normalized.mul(&tangent(1)?)?)?; }
                    if *bias { if let Some(db) = &tangents[2] { result = result.add(db)?; } }
                    Ok(result)
                }
                _ => self.local_reverse_jvp(output, tangents),
            },
        };
        result.map(|tangent| if matches!(self.tag, Some(OpTag::Unary(UnaryKind::Tanh))) { tangent.save_tensor_versions(&[output]) } else { tangent })
    }

    // Opaque first-order operators retain numerical JVP coverage through their
    // own callback. Graph-building uses that same Op's differentiable VJP or
    // reports its existing unsupported boundary; no outer tape is reconstructed.
    fn local_reverse_jvp(&self, output: &Tensor, tangents: &[Option<Tensor>]) -> Result<Tensor> {
        if output.dtype() != DType::F32 || output.device() != Device::Cpu {
            return Err(Error::Unsupported { op: "forward_ad", msg: "opaque local JVP fallback currently supports CPU float32 outputs".into() });
        }
        let graph = crate::autograd::is_grad_enabled() && self.inputs.iter().any(Tensor::requires_grad);
        let count = output.numel();
        let mut rows = Vec::new();
        rows.try_reserve_exact(count).map_err(|_| Error::InvalidShape { op: "forward_ad", msg: "local JVP rows exceed allocation capacity".into() })?;
        for row in 0..count.max(1) {
            let mut basis = crate::typed_math::filled("forward_ad", output.shape(), 0.0f32)?;
            if count != 0 { basis[row] = 1.; }
            let seed = Tensor::from_vec(basis, output.shape())?;
            let contributions = crate::autograd::with_grad_enabled(graph, || self.vjp(&seed, graph, false, None, None, Some(output)))?;
            if contributions.len() != self.inputs.len() { return Err(Error::Unsupported { op: "forward_ad", msg: "local backward gradient arity does not match inputs".into() }); }
            let mut value: Option<Tensor> = None;
            for ((input, contribution), tangent) in self.inputs.iter().zip(contributions).zip(tangents) {
                let Some(contribution) = contribution else { continue; };
                if contribution.shape() != input.shape() { return Err(Error::ShapeMismatch { op: "forward_ad backward", lhs: input.shape().to_vec(), rhs: contribution.shape().to_vec() }); }
                if contribution.dtype() != input.dtype() { return Err(Error::DtypeMismatch { op: "forward_ad backward", expected: input.dtype(), got: contribution.dtype() }); }
                if contribution.device() != input.device() { return Err(Error::DeviceMismatch { op: "forward_ad backward", lhs: input.device(), rhs: contribution.device() }); }
                if let Some(tangent) = tangent {
                    let term = contribution.mul(tangent)?.sum();
                    value = Some(if let Some(previous) = value { previous.add(&term)? } else { term });
                }
            }
            let value = value.unwrap_or_else(|| Tensor::scalar(0.));
            if count == 0 { return Ok(value.reshape(&[1])?.narrow(0, 0, 0)?.reshape(output.shape())?.save_tensor_versions(&self.inputs.iter().collect::<Vec<_>>())); }
            rows.push(value.reshape(&[1])?);
        }
        Ok(Tensor::cat(&rows, 0)?.reshape(output.shape())?.save_tensor_versions(&self.inputs.iter().collect::<Vec<_>>()))
    }
}
