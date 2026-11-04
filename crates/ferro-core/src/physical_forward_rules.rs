//! Experimental physical-batch JVPs at the existing Op recorder.
use super::{ForwardOp, Op};
use crate::dispatch::{BinaryKind, OpTag, UnaryKind};
use crate::{Device, DType, Error, Result, Tensor};

#[path = "physical_forward_compat.rs"]
mod compat;

fn shape(batch: usize, original: &[usize]) -> Result<Vec<usize>> {
    let rank = original.len().checked_add(1).ok_or_else(|| Error::InvalidShape { op: "physical forward AD", msg: "rank overflows".into() })?;
    let mut shape = Vec::new();
    shape.try_reserve_exact(rank).map_err(|_| Error::InvalidShape { op: "physical forward AD", msg: "rank metadata exceeds capacity".into() })?;
    shape.push(batch);
    shape.extend_from_slice(original);
    crate::shape::checked_numel("physical forward AD", &shape)?;
    Ok(shape)
}

fn constant(value: f32, dtype: DType) -> Result<Tensor> {
    if dtype == DType::F64 { crate::typed_math::full_f64(&[], value as f64) }
    else { Tensor::full_on(&[], value, Device::Cpu) }
}

impl Op {
    pub(crate) fn jvp_many_physical(&self, tangents: &[Option<Tensor>], batch: usize) -> Result<Vec<Option<Tensor>>> {
        match self.multi_forward {
            Some(super::MultiForwardOp::PreluBackward) => crate::ops_ext::prelu_backward::jvp_physical(self.inputs(), tangents, batch),
            None => Err(Error::Unsupported { op: "multi-output physical JVP", msg: "generic shared callbacks have no physical forward rule".into() }),
        }
    }

    pub(crate) fn jvp_physical(&self, output: &Tensor, tangents: &[Option<Tensor>], batch: usize) -> Result<Tensor> {
        let inputs = self.inputs();
        let a = &inputs[0];
        let target = shape(batch, output.shape())?;
        let tangent = |index: usize| -> Result<Tensor> {
            tangents[index].as_ref().cloned().map(Ok).unwrap_or_else(|| Tensor::structural_zero(inputs[index].shape(), inputs[index].dtype(), inputs[index].device()))
        };
        let aligned = |index: usize| -> Result<Tensor> {
            let value = tangent(index)?;
            if tangents[index].is_none() || inputs[index].ndim() == output.ndim() { return Ok(value); }
            let mut logical = vec![1; output.ndim() + 1];
            logical[0] = batch;
            logical[output.ndim() + 1 - inputs[index].ndim()..].copy_from_slice(inputs[index].shape());
            value.view(&logical)
        };
        let expand = |value: Tensor| -> Result<Tensor> { if value.shape() == target { Ok(value) } else { value.expand(&target) } };
        let g = || tangent(0);
        let result = match self.tag {
            Some(OpTag::Binary(kind)) => {
                let b = &inputs[1];
                match kind {
                    BinaryKind::Add => match (&tangents[0], &tangents[1]) {
                        (Some(_), Some(_)) => aligned(0)?.add(&aligned(1)?),
                        (Some(_), None) => expand(aligned(0)?),
                        (None, Some(_)) => expand(aligned(1)?),
                        _ => unreachable!("active input tangent"),
                    },
                    BinaryKind::Sub => match (&tangents[0], &tangents[1]) {
                        (Some(_), Some(_)) => aligned(0)?.sub(&aligned(1)?),
                        (Some(_), None) => expand(aligned(0)?),
                        (None, Some(_)) => expand(aligned(1)?.neg()),
                        _ => unreachable!("active input tangent"),
                    },
                    BinaryKind::Mul => aligned(0)?.mul(b)?.add(&aligned(1)?.mul(a)?),
                    BinaryKind::Div => aligned(0)?.sub(&aligned(1)?.mul(&a.div(b)?)?)?.div(b),
                    BinaryKind::ReluBackward => crate::ops::relu_backward(&expand(aligned(0)?)?, &b.expand(&target)?),
                }
            }
            Some(OpTag::Unary(UnaryKind::Neg)) => Ok(g()?.neg()),
            Some(OpTag::Unary(UnaryKind::Exp)) => g()?.mul(output),
            Some(OpTag::Unary(UnaryKind::Log)) => g()?.div(a),
            Some(OpTag::Unary(UnaryKind::Powf(power))) => {
                if power == 0. { if output.dtype() == DType::F64 { crate::typed_math::full_f64(&target, 0.) } else { Tensor::full_on(&target, 0., Device::Cpu) } }
                else { g()?.mul(&a.powf(power - 1.))?.mul(&constant(power, a.dtype())?) }
            }
            Some(OpTag::Unary(UnaryKind::Relu)) => crate::ops::relu_backward(&g()?, &output.expand(&target)?),
            Some(OpTag::Unary(UnaryKind::Abs)) => g()?.mul(&a.sign()?),
            Some(OpTag::Unary(UnaryKind::Sign | UnaryKind::Gtz)) => if g()?.dtype() == DType::F64 { crate::typed_math::full_f64(&target, 0.) } else { Tensor::full_on(&target, 0., Device::Cpu) },
            Some(OpTag::Unary(UnaryKind::Clamp { min, max })) => {
                let mask = a.ge_tensor(&constant(min, a.dtype())?)?.logical_and(&a.le_tensor(&constant(max, a.dtype())?)?)?;
                let direction = g()?;
                Tensor::where_cond(&mask, &direction, &constant(0., direction.dtype())?)
            }
            Some(OpTag::Unary(UnaryKind::Sqrt)) => g()?.div(&output.mul(&constant(2., output.dtype())?)?),
            Some(OpTag::Unary(UnaryKind::Tanh | UnaryKind::Sigmoid)) => {
                let kind = if matches!(self.tag, Some(OpTag::Unary(UnaryKind::Tanh))) { crate::ops_ext::activation_backward::Kind::Tanh } else { crate::ops_ext::activation_backward::Kind::Sigmoid };
                crate::ops_ext::activation_backward::backward(kind, &g()?, &output.expand(&target)?)
            }
            Some(OpTag::Unary(UnaryKind::Silu)) => {
                let s = a.sigmoid();
                g()?.mul(&s.add(&a.mul(&s)?.mul(&constant(1., a.dtype())?.sub(&s)?)?)?)
            }
            _ => match &self.forward {
                Some(ForwardOp::SigmoidBackward | ForwardOp::TanhBackward) => {
                    let y = &inputs[1];
                    let kind = if matches!(self.forward, Some(ForwardOp::TanhBackward)) { crate::ops_ext::activation_backward::Kind::Tanh } else { crate::ops_ext::activation_backward::Kind::Sigmoid };
                    let first = crate::ops_ext::activation_backward::backward(kind, &g()?, &if tangents[0].is_some() { y.expand(&target)? } else { y.clone() })?;
                    let factor = y.mul(&constant(-2., y.dtype())?)?;
                    let second = match kind {
                        crate::ops_ext::activation_backward::Kind::Tanh => tangent(1)?.mul(&factor.mul(a)?)?,
                        crate::ops_ext::activation_backward::Kind::Sigmoid => tangent(1)?.mul(a)?.mul(&factor.add(&constant(1., y.dtype())?)?)?,
                    };
                    first.add(&second)
                }
                Some(ForwardOp::Alias) => g(),
                Some(ForwardOp::CumExtrema { dim, .. }) => {
                    use crate::ops_ext::indexed::IndexMap;
                    if a.ndim() == 0 {
                        let index = inputs[1].expand(&[batch, 1])?;
                        g()?.unsqueeze(1)?.indexed_gather(&index, IndexMap::Axis(1), true)?.squeeze(1)
                    } else {
                        let index = inputs[1].expand(&shape(batch, inputs[1].shape())?)?;
                        g()?.indexed_gather(&index, IndexMap::Axis(dim + 1), true)
                    }
                }
                Some(ForwardOp::Prod { dim, keepdim, .. }) => compat::product(a, output, &g()?, *dim, *keepdim, batch),
                Some(ForwardOp::Prelu) => compat::prelu(inputs, tangents, batch),
                Some(ForwardOp::IndexedGather { map, kind, save_input }) => {
                    use crate::ops_ext::indexed::IndexMap;
                    let shifted = match map {
                        IndexMap::Axis(axis) if a.ndim() != 0 => IndexMap::Axis(axis + 1),
                        IndexMap::Select { axis, index_dims } => IndexMap::Select { axis: axis + 1, index_dims: *index_dims },
                        _ => return self.local_reverse_jvp_physical(output, tangents, batch),
                    };
                    let index = inputs[1].expand(&shape(batch, inputs[1].shape())?)?;
                    g()?.indexed_gather_kind(&index, shifted, *kind, *save_input)
                }
                Some(ForwardOp::IndexedScatter { map, kind, shape: original, save_input, strides }) => {
                    use crate::ops_ext::indexed::IndexMap;
                    let shifted = match map {
                        IndexMap::Axis(axis) if !original.is_empty() => IndexMap::Axis(axis + 1),
                        IndexMap::Select { axis, index_dims } => IndexMap::Select { axis: axis + 1, index_dims: *index_dims },
                        _ => return self.local_reverse_jvp_physical(output, tangents, batch),
                    };
                    if strides.is_some() { return self.local_reverse_jvp_physical(output, tangents, batch); }
                    let index = inputs[1].expand(&shape(batch, inputs[1].shape())?)?;
                    Tensor::indexed_scatter_saved(&g()?, &index, &shape(batch, original)?, shifted, *kind, *save_input, None)
                }
                Some(ForwardOp::WhereConditional) => {
                    let value = |i: usize| -> Result<Tensor> {
                        if tangents[i].is_some() { aligned(i) }
                        else { constant(0., inputs[i].dtype())?.expand(inputs[i].shape()) }
                    };
                    let (left,right) = (value(1)?,value(2)?);
                    let dtype = crate::typed_math::result_type(&left,&right)?;
                    let prepare = |value: Tensor| -> Result<Tensor> {
                        let value = if output.numel() == 0 { value.expand(&target)? } else { value };
                        value.to_dtype_autograd(dtype,false)
                    };
                    Tensor::where_cond(&inputs[0], &prepare(left)?, &prepare(right)?)
                }
                Some(ForwardOp::AddAlpha { alpha, subtract }) => {
                    alpha.validate_ad()?;
                    let prepared = |index: usize| -> Result<Option<Tensor>> {
                        tangents[index].as_ref().map(|_| {
                            let value = aligned(index)?;
                            let value = if output.numel() == 0 { expand(value)? } else { value };
                            if inputs[index].dtype() == output.dtype() { Ok(value) }
                            else { value.to_dtype_autograd(output.dtype(),false) }
                        }).transpose()
                    };
                    match (prepared(0)?,prepared(1)?) {
                        (Some(left),Some(right)) => { let scaled = alpha.scale(&right)?; if *subtract { left.sub(&scaled) } else { left.add(&scaled) } },
                        (Some(value),None) => expand(value),
                        (None,Some(value)) => { let scaled = alpha.scale(&value)?; expand(if *subtract { scaled.neg() } else { scaled }) },
                        _ => unreachable!("active input tangent"),
                    }
                }
                Some(ForwardOp::RoundedDiv(_)) => crate::ops_ext::div_rounding::zero_like(output)?.expand(&target),
                Some(ForwardOp::FloorDivide) => Err(Error::Unsupported { op: "floor_divide forward_ad", msg: "forward AD for floor_divide is not implemented".into() }),
                Some(ForwardOp::CumProd { .. }) if a.ndim() == 0 => {
                    let input = a.expand(&[batch, 1])?;
                    let output = output.expand(&[batch, 1])?;
                    crate::ops_ext::cumprod::jvp(&input, &output, &g()?.unsqueeze(1)?, 1)?.squeeze(1)
                }
                Some(ForwardOp::CumProd { dim, .. }) if a.ndim() != 0 => crate::ops_ext::cumprod::jvp(&a.expand(&shape(batch,a.shape())?)?, &output.expand(&target)?, &g()?, dim + 1),
                Some(ForwardOp::ScalarPower(power)) => {
                    if *power == 0. { output.zeros_like_layout()?.expand(&target) }
                    else { crate::ops_ext::powf::scalar_power_backward(&g()?, a, *power) }
                }
                Some(ForwardOp::Square) => g()?.mul(&a.powf(1.).mul(&constant(2., a.dtype())?)?),
                Some(ForwardOp::Frac) => g(),
                Some(ForwardOp::Rounding(_)) => crate::ops_ext::div_rounding::zero_like(&g()?),
                Some(ForwardOp::Rsqrt) => g()?.mul(&constant(-0.5, g()?.dtype())?)?.mul(&output.mul(output)?.mul(output)?),
                Some(ForwardOp::Sum | ForwardOp::Mean) => {
                    let mut value = g()?;
                    if a.numel() == 0 { value = value.reshape(&[batch,0])?.sum_dim(1,false)?; }
                    else { for _ in 0..a.ndim() { value = value.sum_dim(1, false)?; } }
                    if matches!(self.forward, Some(ForwardOp::Mean)) { value.div(&constant(a.numel() as f32, value.dtype())?) }
                    else if a.ndim() == 0 { value.clone_tensor(false) } else { Ok(value) }
                }
                Some(ForwardOp::SumDim(dim, keep)) => if a.ndim() == 0 { g()?.clone_tensor(false) } else { g()?.sum_dim(dim + 1, *keep) },
                Some(ForwardOp::CumSum { dim, dtype, reverse }) => if a.ndim() == 0 { g()?.to_dtype_autograd(dtype.unwrap_or(g()?.dtype()), true) } else { g()?.cumulative_sum(dim + 1, *dtype, *reverse) },
                Some(ForwardOp::Reshape(original)) => g()?.reshape(&shape(batch, original)?),
                Some(ForwardOp::View(original)) => g()?.view(&shape(batch, original)?),
                Some(ForwardOp::Squeeze(dim)) => g()?.squeeze(dim + 1),
                Some(ForwardOp::Unsqueeze(dim)) => g()?.unsqueeze(dim + 1),
                Some(ForwardOp::Transpose(left, right)) => g()?.transpose(left + 1, right + 1),
                Some(ForwardOp::Permute(dims)) => {
                    let mut permutation = vec![0];
                    permutation.extend(dims.iter().map(|dim| dim + 1));
                    g()?.permute(&permutation)
                }
                Some(ForwardOp::Expand(original)) => {
                    let mut aligned = vec![1; original.len() + 1]; aligned[0] = batch;
                    aligned[original.len() + 1 - a.ndim()..].copy_from_slice(a.shape());
                    g()?.view(&aligned)?.expand(&target)
                }
                Some(ForwardOp::Cast(dtype)) => g()?.to_dtype_autograd(*dtype, true),
                Some(ForwardOp::CopyDualLayout { strides, offset, storage_len }) => {
                    let count = batch.checked_mul(*storage_len).ok_or_else(|| Error::InvalidShape { op: "physical forward AD", msg: "copied batch storage overflows".into() })?;
                    let mut layout = vec![*storage_len]; layout.extend_from_slice(strides);
                    g()?.copy_dual_layout(&layout,*offset,count)
                }
                Some(ForwardOp::Clone { preserve_layout }) => g()?.clone_tensor(*preserve_layout),
                Some(ForwardOp::Slice { dim, start, end, step }) => g()?.slice(dim + 1, *start, *end, *step),
                Some(ForwardOp::SliceAdjoint { shape: original, dim, start, end, step }) => g()?.slice_adjoint(&shape(batch, original)?, dim + 1, *start, *end, *step),
                Some(ForwardOp::Sin) => g()?.mul(&a.cos()?),
                Some(ForwardOp::Cos) => g()?.neg().mul(&a.sin()?),
                Some(ForwardOp::Cat(dim)) => {
                    let dtype = if tangents.iter().flatten().any(|value| value.dtype() == DType::F64) { DType::F64 } else { output.dtype() };
                    let values = (0..inputs.len()).map(|index| {
                        let value = tangent(index)?.to_dtype_autograd(dtype, false)?;
                        if tangents[index].is_none() { value.expand(&shape(batch, inputs[index].shape())?) } else { Ok(value) }
                    }).collect::<Result<Vec<_>>>()?;
                    Tensor::cat(&values, dim + 1)
                }
                Some(ForwardOp::Softmax(dim)) => {
                    let g = g()?;
                    output.mul(&g.sub(&g.mul(output)?.sum_dim(dim + 1, true)?)?)
                }
                Some(ForwardOp::MatMul) => {
                    let b = &inputs[1];
                    let first = if let Some(ga) = &tangents[0] { ga.bmm(&b.expand(&shape(batch, b.shape())?)?)? } else { tangent(0)?.matmul(b)? };
                    let second = if let Some(gb) = &tangents[1] { a.expand(&shape(batch, a.shape())?)?.bmm(gb)? } else { a.matmul(&tangent(1)?)? };
                    first.add(&second)
                }
                Some(ForwardOp::Bmm) => {
                    let b = &inputs[1];
                    let count = batch.checked_mul(a.shape()[0]).ok_or_else(|| Error::InvalidShape { op: "physical forward AD", msg: "flattened batch overflows".into() })?;
                    let first = if let Some(ga) = &tangents[0] {
                        ga.reshape(&[count,a.shape()[1],a.shape()[2]])?.bmm(&b.expand(&shape(batch,b.shape())?)?.reshape(&[count,b.shape()[1],b.shape()[2]])?)?.reshape(&target)?
                    } else { tangent(0)?.bmm(b)? };
                    let second = if let Some(gb) = &tangents[1] {
                        a.expand(&shape(batch,a.shape())?)?.reshape(&[count,a.shape()[1],a.shape()[2]])?.bmm(&gb.reshape(&[count,b.shape()[1],b.shape()[2]])?)?.reshape(&target)?
                    } else { a.bmm(&tangent(1)?)? };
                    first.add(&second)
                }
                Some(ForwardOp::LayerNorm { weight, bias, .. }) => {
                    let [mean, invstd] = self.saved_jvp.as_slice() else { return Err(Error::Unsupported { op: "layer norm JVP", msg: "original saved statistics are unavailable".into() }); };
                    let dim = a.ndim() - 1;
                    let width = constant(a.shape()[dim] as f32, a.dtype())?;
                    let centered = a.sub(mean)?;
                    let normalized = centered.mul(invstd)?;
                    let da = g()?;
                    let axis = dim + usize::from(tangents[0].is_some());
                    let centered_tangent = da.sub(&da.sum_dim(axis,true)?.div(&width)?)?;
                    let invstd_tangent = invstd.powf(3.).neg().mul(&centered_tangent)?.mul(&centered)?.sum_dim(axis,true)?.div(&width)?;
                    let mut result = centered_tangent.mul(invstd)?.add(&centered.mul(&invstd_tangent)?)?;
                    if *weight { result = result.mul(&inputs[1])?.add(&normalized.mul(&aligned(1)?)?)?; }
                    if *bias && tangents[2].is_some() { result = result.add(&aligned(2)?)?; }
                    Ok(result)
                }
                _ => self.local_reverse_jvp_physical(output, tangents, batch),
            },
        };
        result.map(|tangent| if matches!(self.tag, Some(OpTag::Unary(UnaryKind::Tanh))) { tangent.save_tensor_versions(&[output]) } else { tangent })
    }

    // The same Op's callback supplies only the local Jacobian; its contraction
    // keeps the physical seed axis. This retains the existing numerical fallback,
    // without claiming native batched layout/provenance for an opaque operation.
    fn local_reverse_jvp_physical(&self, output: &Tensor, tangents: &[Option<Tensor>], batch: usize) -> Result<Tensor> {
        if output.dtype() != DType::F32 || output.device() != Device::Cpu {
            return Err(Error::Unsupported { op: "physical forward AD", msg: "opaque local JVP fallback currently supports CPU float32 outputs".into() });
        }
        let graph = crate::autograd::is_grad_enabled() && self.inputs.iter().any(Tensor::requires_grad);
        let count = output.numel();
        let target = shape(batch,output.shape())?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(count).map_err(|_| Error::InvalidShape { op: "physical forward AD", msg: "local JVP rows exceed allocation capacity".into() })?;
        for row in 0..count.max(1) {
            let mut basis = crate::typed_math::filled("physical forward AD",output.shape(),0.0f32)?;
            if count != 0 { basis[row] = 1.; }
            let seed = Tensor::from_vec(basis,output.shape())?;
            let contributions = crate::autograd::with_grad_enabled(graph, || self.vjp(&seed,graph,false,None,None,Some(output)))?;
            if contributions.len() != self.inputs.len() { return Err(Error::Unsupported { op: "physical forward AD", msg: "local backward gradient arity does not match inputs".into() }); }
            let mut value: Option<Tensor> = None;
            for ((input,contribution),tangent) in self.inputs.iter().zip(contributions).zip(tangents) {
                let Some(contribution) = contribution else { continue; };
                if contribution.shape() != input.shape() { return Err(Error::ShapeMismatch { op: "physical forward AD backward", lhs: input.shape().to_vec(), rhs: contribution.shape().to_vec() }); }
                if contribution.dtype() != input.dtype() { return Err(Error::DtypeMismatch { op: "physical forward AD backward", expected: input.dtype(), got: contribution.dtype() }); }
                if contribution.device() != input.device() { return Err(Error::DeviceMismatch { op: "physical forward AD backward", lhs: input.device(), rhs: contribution.device() }); }
                if let Some(tangent) = tangent {
                    let mut term = contribution.mul(tangent)?;
                    if input.numel() == 0 { term = term.reshape(&[batch,0])?.sum_dim(1,false)?; }
                    else { for _ in 0..input.ndim() { term = term.sum_dim(1,false)?; } }
                    value = Some(if let Some(previous) = value { previous.add(&term)? } else { term });
                }
            }
            let value = match value { Some(value) => value, None => Tensor::full_on(&[batch],0.,Device::Cpu)? };
            if count == 0 { return Ok(value.unsqueeze(1)?.narrow(1,0,0)?.reshape(&target)?.save_tensor_versions(&self.inputs.iter().collect::<Vec<_>>())); }
            rows.push(value.unsqueeze(1)?);
        }
        Ok(Tensor::cat(&rows,1)?.reshape(&target)?.save_tensor_versions(&self.inputs.iter().collect::<Vec<_>>()))
    }

}
