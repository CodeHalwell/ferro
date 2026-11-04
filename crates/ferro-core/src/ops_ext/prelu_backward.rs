//! The two full-shape PReLU adjoints share one operation and optional seed list.
use crate::{DType, Device, Error, Result, Tensor};
use crate::autograd::MultiForwardOp;

const OP: &str = "prelu backward pair";

fn validate(inputs: &[Tensor]) -> Result<()> {
    if inputs.len() != 3 { return Err(Error::InvalidShape { op: OP, msg: "expected gradient, input and expanded weight".into() }); }
    let x = &inputs[1];
    if !matches!(x.dtype(), DType::F32 | DType::F64) { return Err(Error::DtypeMismatch { op: OP, expected: DType::F32, got: x.dtype() }); }
    crate::typed_math::check(OP, &[&inputs[0], x, &inputs[2]], x.dtype())?;
    for value in inputs {
        if value.shape() != x.shape() { return Err(Error::ShapeMismatch { op: OP, lhs: x.shape().to_vec(), rhs: value.shape().to_vec() }); }
    }
    Ok(())
}

fn dense(shape: &[usize], stride: &[usize], axes: &[usize]) -> bool {
    if shape.contains(&0) { return true; }
    let mut next = 1;
    axes.iter().all(|&axis| {
        if shape[axis] == 1 { return true; }
        if stride[axis] != next { return false; }
        next *= shape[axis]; true
    })
}

// TensorIterator orders the original input, weight, then incoming gradient.
fn raw_layout(x: &Tensor, w: &Tensor, g: &Tensor) -> Vec<usize> {
    let shape = x.shape(); let inputs = [x, w, g];
    let mut axes = (0..shape.len()).rev().collect::<Vec<_>>();
    if inputs.iter().all(|v| dense(shape, v.strides(), &axes)) { return crate::shape::default_strides(shape); }
    if shape.len() == 4 && inputs.iter().all(|v| dense(shape, v.strides(), &[1, 3, 2, 0])) {
        let mut out = vec![0; 4]; let mut next = 1;
        for axis in [1, 3, 2, 0] { out[axis] = next; next *= shape[axis].max(1); }
        return out;
    }
    let mut common = (0..shape.len()).collect::<Vec<_>>(); common.sort_by_key(|&axis| x.strides()[axis]);
    if inputs.iter().all(|v| v.strides() == x.strides() && dense(shape, v.strides(), &common)) { return x.strides().to_vec(); }
    for index in 1..axes.len() {
        let mut current = index;
        for previous in (0..index).rev() {
            let (left, right) = (axes[previous], axes[current]); let mut order = 0;
            for value in inputs {
                let (ls, rs) = (value.strides()[left], value.strides()[right]);
                if ls == 0 || rs == 0 { continue; }
                if ls < rs { order = -1; break; }
                if ls > rs || shape[left] > shape[right] { order = 1; break; }
            }
            if order < 0 { break; }
            if order > 0 { axes.swap(previous, current); current = previous; }
        }
    }
    if axes.iter().copied().eq((0..shape.len()).rev()) { return crate::shape::default_strides(shape); }
    let mut stride = vec![0; shape.len()]; let mut next = 1;
    for axis in axes { stride[axis] = next; next *= shape[axis]; }
    stride
}

fn owning_layout(raw: Tensor, shape: &[usize], stride: Vec<usize>) -> Tensor {
    Tensor::from_parts(raw.0.storage.clone(), shape.to_vec(), stride, 0, Device::Cpu, false, None)
}

fn zero(dtype: DType) -> Result<Tensor> {
    if dtype == DType::F64 { Tensor::from_vec_f64(vec![0.], &[]) } else { Ok(Tensor::scalar(0.)) }
}

fn zeros_like(input: &Tensor) -> Result<Tensor> {
    let stride = crate::typed_math::copy_pointwise_layout(input);
    let raw = if input.dtype() == DType::F64 {
        Tensor::from_vec_f64(crate::typed_math::filled(OP, input.shape(), 0.0f64)?, input.shape())?
    } else { Tensor::from_vec(crate::typed_math::filled(OP, input.shape(), 0.0f32)?, input.shape())? };
    Ok(owning_layout(raw, input.shape(), stride))
}

pub(crate) fn backward(g: &Tensor, x: &Tensor, w: &Tensor) -> Result<Vec<Tensor>> {
    let inputs = vec![g.clone(), x.clone(), w.clone()]; validate(&inputs)?;
    let stride = raw_layout(x, w, g);
    let sources = [g, x, w].map(|v| crate::typed_math::pointwise_source(v, x.shape(), &stride)).into_iter().collect::<Result<Vec<_>>>()?;
    macro_rules! compute { ($read:ident, $ctor:ident, $zero:expr) => {{
        let grad = sources[0].$read(); let values = sources[1].$read(); let slope = sources[2].$read();
        let mut gx = crate::typed_math::filled(OP, x.shape(), $zero)?;
        let mut gw = crate::typed_math::filled(OP, x.shape(), $zero)?;
        for (((left, right), (&g, &x)), &w) in gx.iter_mut().zip(&mut gw).zip(grad.iter().zip(&values)).zip(&slope) {
            *left = if x > 0. { g } else { g * w };
            *right = if x > 0. { $zero } else { g * x };
        }
        vec![owning_layout(Tensor::$ctor(gx, x.shape())?, x.shape(), stride.clone()), owning_layout(Tensor::$ctor(gw, x.shape())?, x.shape(), stride)]
    }}; }
    let outputs = if x.dtype() == DType::F64 { compute!(to_vec_f64, from_vec_f64, 0.0f64) } else { compute!(to_vec, from_vec, 0.0f32) };
    let saved = inputs.clone();
    Tensor::record_many_forward_optional(outputs, inputs, &[true, true], MultiForwardOp::PreluBackward,
        move |seeds| graph_vjp(&saved, seeds).expect("validated PReLU pair adjoint"))
}

fn axes(input: &Tensor) -> Result<(Vec<usize>, Vec<usize>)> {
    let stride = crate::typed_math::unary_pointwise_layout(input);
    let mut order = (0..input.ndim()).collect::<Vec<_>>();
    if stride != crate::shape::default_strides(input.shape()) { order.sort_by(|&a, &b| stride[b].cmp(&stride[a]).then_with(|| b.cmp(&a))); }
    let shape = order.iter().map(|&axis| input.shape()[axis]).collect::<Vec<_>>();
    if order.iter().zip(crate::shape::default_strides(&shape)).any(|(&axis, expected)| stride[axis] != expected) {
        return Err(Error::Unsupported { op: OP, msg: "this singleton stride order is not representable by a dense permutation".into() });
    }
    let mut inverse = vec![0; order.len()]; for (physical, &logical) in order.iter().enumerate() { inverse[logical] = physical; }
    Ok((order, inverse))
}

fn select(condition: &Tensor, left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let dtype = crate::typed_math::result_type(left, right)?;
    Tensor::where_cond(condition, &left.to_dtype_autograd(dtype, false)?, &right.to_dtype_autograd(dtype, false)?)
}

pub(crate) fn graph_vjp(inputs: &[Tensor], seeds: &[Option<Tensor>]) -> Result<Vec<Option<Tensor>>> {
    validate(inputs)?;
    if seeds.len() != 2 { return Err(Error::InvalidShape { op: OP, msg: "expected two optional output cotangents".into() }); }
    if seeds.iter().all(Option::is_none) { return Ok(vec![None, None, None]); }
    for seed in seeds.iter().flatten() {
        crate::typed_math::check(OP, &[seed], inputs[0].dtype())?;
        if seed.shape() != inputs[0].shape() { return Err(Error::ShapeMismatch { op: OP, lhs: inputs[0].shape().to_vec(), rhs: seed.shape().to_vec() }); }
    }
    let (order, inverse) = axes(&inputs[1])?;
    let g = inputs[0].permute(&order)?; let x = inputs[1].permute(&order)?; let w = inputs[2].permute(&order)?;
    let a = seeds[0].as_ref().map(|v| v.permute(&order)).transpose()?;
    let b = seeds[1].as_ref().map(|v| v.permute(&order)).transpose()?;
    let zeros = zero(x.dtype())?; let positive = x.ge_tensor(&zeros)?;
    let dg = match (&a, &b) {
        (Some(a), Some(b)) => select(&positive, a, &a.mul(&w)?.add(&b.mul(&x)?)?)?,
        (Some(a), None) => select(&positive, a, &a.mul(&w)?)?,
        (None, Some(b)) => select(&positive, &zeros, &b.mul(&x)?)?,
        (None, None) => unreachable!(),
    }.permute(&inverse)?;
    let dx = if let Some(b) = b { select(&positive, &zeros, &g.mul(&b)?)?.permute(&inverse)? } else { zeros_like(&inputs[1])? };
    let dw = if let Some(a) = a { select(&positive, &zeros, &g.mul(&a)?)?.permute(&inverse)? } else { zeros_like(&inputs[2])? };
    Ok(vec![Some(dg), Some(dx), Some(dw)])
}

pub(crate) fn jvp(inputs: &[Tensor], tangents: &[Option<Tensor>]) -> Result<Vec<Option<Tensor>>> {
    validate(inputs)?;
    if tangents.len() != 3 { return Err(Error::InvalidShape { op: OP, msg: "expected three input tangent slots".into() }); }
    for (input, direction) in inputs.iter().zip(tangents) {
        if let Some(direction) = direction {
            if !matches!(direction.dtype(), DType::F32 | DType::F64) { return Err(Error::DtypeMismatch { op: OP, expected: DType::F32, got: direction.dtype() }); }
            crate::typed_math::check(OP, &[direction], direction.dtype())?;
            if direction.shape() != input.shape() { return Err(Error::ShapeMismatch { op: OP, lhs: input.shape().to_vec(), rhs: direction.shape().to_vec() }); }
        }
    }
    if tangents.iter().all(Option::is_none) { return Ok(vec![None, None]); }
    let (order, inverse) = axes(&inputs[1])?;
    let values = inputs.iter().map(|v| v.permute(&order)).collect::<Result<Vec<_>>>()?;
    let directions = inputs.iter().zip(tangents).map(|(input, direction)| match direction {
        Some(value) => value.permute(&order),
        None => Tensor::structural_zero(&[], input.dtype(), Device::Cpu)?.expand(input.shape())?.permute(&order),
    }).collect::<Result<Vec<_>>>()?;
    let (g, x, w) = (&values[0], &values[1], &values[2]);
    let (dg, dx, dw) = (&directions[0], &directions[1], &directions[2]);
    let positive = x.ge_tensor(&zero(x.dtype())?)?;
    let first = if tangents[0].is_some() || tangents[2].is_some() {
        let negative = dg.mul(w)?.add(&g.mul(dw)?)?;
        Some(select(&positive, dg, &negative)?.permute(&inverse)?)
    } else { None };
    let second = if tangents[0].is_some() || tangents[1].is_some() {
        let negative = g.mul(dx)?.add(&dg.mul(x)?)?;
        Some(select(&positive, &zero(negative.dtype())?, &negative)?.permute(&inverse)?)
    } else { None };
    Ok(vec![first, second])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(values: &[f64], shape: &[usize], dtype: DType, tracked: bool) -> Tensor {
        Tensor::from_vec_f64(values.to_vec(), shape).unwrap().to_dtype(dtype).requires_grad_(tracked).unwrap()
    }
    fn check(value: &Tensor, expected: &[f64]) {
        assert_eq!(value.numel(), expected.len());
        for (actual, &expected) in value.to_vec_f64().iter().zip(expected) {
            assert!(actual.is_nan() && expected.is_nan() || *actual == expected || (actual - expected).abs() < 2e-6, "{actual} != {expected}");
            if expected == 0. { assert_eq!(actual.is_sign_negative(), expected.is_sign_negative()); }
        }
    }

    #[test]
    fn finite_difference_checks_every_raw_pair_input_through_both_siblings() {
        let inputs=[tensor(&[0.4,-0.2,0.8,0.6],&[4],DType::F32,false),tensor(&[-0.7,1.2,-1.1,0.3],&[4],DType::F32,false),tensor(&[0.3,0.5,-0.6,0.2],&[4],DType::F32,false)];
        crate::testkit::grad_check_strict(&inputs,|values| {
            let pair=backward(&values[0],&values[1],&values[2]).unwrap();
            pair[0].sum().add(&pair[1].square().unwrap().sum()).unwrap()
        });
    }

    #[test]
    fn raw_pair_and_joint_derivative_preserve_boundaries_and_nonfinite_branches() {
        for dtype in [DType::F32, DType::F64] {
            let g = tensor(&[2.,3.,4.,5.,6.,7.,8.], &[7], dtype, true);
            let x = tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN], &[7], dtype, true);
            let w = tensor(&[f64::INFINITY;7], &[7], dtype, true);
            let output = backward(&g,&x,&w).unwrap();
            check(&output[0],&[f64::INFINITY,f64::INFINITY,f64::INFINITY,f64::INFINITY,6.,7.,f64::INFINITY]);
            check(&output[1],&[f64::NEG_INFINITY,-6.,-0.,0.,0.,0.,f64::NAN]);
            let a=tensor(&[1.;7],&[7],dtype,false);let b=tensor(&[2.;7],&[7],dtype,false);
            let result=Tensor::vjp_many_wrt_optional(&[&output[0],&output[1]],&[&g,&x,&w],&[&a,&b],false).unwrap();
            check(result[0].as_ref().unwrap(),&[f64::NAN,f64::INFINITY,1.,1.,1.,1.,f64::NAN]);
            check(result[1].as_ref().unwrap(),&[4.,6.,0.,0.,0.,0.,16.]);
            check(result[2].as_ref().unwrap(),&[2.,3.,0.,0.,0.,0.,8.]);
            assert!(g.grad().is_none()&&x.grad().is_none()&&w.grad().is_none());
        }
    }

    #[test]
    fn absent_sibling_and_shared_signed_zero_differ_from_separate_contributions() {
        for dtype in [DType::F32,DType::F64] {
            let g=tensor(&[1.],&[1],dtype,true);let x=tensor(&[2.],&[1],dtype,true);let w=tensor(&[0.3],&[1],dtype,true);
            let output=backward(&g,&x,&w).unwrap();let a=tensor(&[-0.],&[1],dtype,false);let b=tensor(&[0.],&[1],dtype,false);
            let joint=Tensor::vjp_many_wrt_optional(&[&output[0],&output[1]],&[&g],&[&a,&b],false).unwrap().remove(0).unwrap();
            check(&joint,&[-0.]);
            let separate=output[0].vjp_wrt(&[&g],&a,false).unwrap()[0].add(&output[1].vjp_wrt(&[&g],&b,false).unwrap()[0]).unwrap();
            check(&separate,&[0.]);
            let x=tensor(&[f64::NEG_INFINITY],&[1],dtype,true);let w=tensor(&[f64::INFINITY],&[1],dtype,true);
            let output=backward(&g,&x,&w).unwrap();let one=tensor(&[1.],&[1],dtype,false);
            check(&output[0].vjp_wrt(&[&g],&one,false).unwrap()[0],&[f64::INFINITY]);
            let joint=Tensor::vjp_many_wrt_optional(&[&output[0],&output[1]],&[&g],&[&one,&b],false).unwrap();
            check(joint[0].as_ref().unwrap(),&[f64::NAN]);
        }
    }

    #[test]
    fn graph_pair_preserves_live_seed_dependencies_and_defined_absent_zeros() {
        let g=tensor(&[0.7],&[1],DType::F32,true);let x=tensor(&[-2.],&[1],DType::F32,true);let w=tensor(&[0.3],&[1],DType::F32,true);
        let a=tensor(&[1.1],&[1],DType::F32,true);let b=tensor(&[-0.4],&[1],DType::F32,true);
        let output=backward(&g,&x,&w).unwrap();
        let result=Tensor::vjp_many_wrt_optional(&[&output[0],&output[1]],&[&g,&x,&w],&[&a,&b],true).unwrap();
        for (gradient,value) in result.iter().zip([1.13,-0.28,0.77]) { check(gradient.as_ref().unwrap(),&[value]); }
        let next=result[0].as_ref().unwrap().sum().vjp_wrt_optional(&[&g,&x,&w,&a,&b],&Tensor::scalar(1.),false).unwrap();
        assert!(next[0].is_none());
        for (gradient,value) in next[1..].iter().zip([-0.4,1.1,0.3,-2.]) { check(gradient.as_ref().unwrap(),&[value]); }
        let only=output[0].vjp_wrt_optional(&[&g,&x,&w],&a,true).unwrap();
        check(only[1].as_ref().unwrap(),&[0.]);assert!(!only[1].as_ref().unwrap().requires_grad());
        let only=output[1].vjp_wrt_optional(&[&g,&x,&w],&b,true).unwrap();
        check(only[2].as_ref().unwrap(),&[0.]);assert!(!only[2].as_ref().unwrap().requires_grad());
        assert!(graph_vjp(&[g,x,w],&[None,None]).unwrap().iter().all(Option::is_none));
    }

    #[test]
    fn raw_three_input_priority_and_graph_predicate_layout_are_distinct() {
        let plain=tensor(&[1.,2.,3.,4.,5.,6.],&[2,3],DType::F32,true);
        let transposed=tensor(&[-1.,2.,-3.,4.,-5.,6.],&[3,2],DType::F32,true).transpose(0,1).unwrap();
        let broadcast=tensor(&[-1.,2.,-3.],&[1,3],DType::F32,true).expand(&[2,3]).unwrap();
        for (x,w,g,expected) in [(&transposed,&plain,&plain,vec![1,2]),(&broadcast,&transposed,&plain,vec![1,2]),(&broadcast,&plain,&transposed,vec![3,1])] {
            let output=backward(g,x,w).unwrap();for value in output {assert_eq!(value.strides(),expected);}
        }
        let seed=Tensor::ones(&[2,3]).requires_grad_(true).unwrap();
        let gradients=graph_vjp(&[plain.clone(),transposed.clone(),plain.clone()],&[Some(seed.clone()),None]).unwrap();
        assert_eq!(gradients[0].as_ref().unwrap().strides(),&[1,2]);
        assert_eq!(gradients[1].as_ref().unwrap().strides(),&[1,2]);
        assert!(!gradients[1].as_ref().unwrap().requires_grad());
        let gradients=graph_vjp(&[plain.clone(),transposed,plain],&[None,Some(seed)]).unwrap();
        assert_eq!(gradients[2].as_ref().unwrap().strides(),&[3,1]);
        assert!(!gradients[2].as_ref().unwrap().requires_grad());
    }

    #[test]
    fn original_inputs_are_saved_while_both_mutable_outputs_are_not() {
        for flags in 1..8 { for changed in 0..3 {
            let inputs=[0.7,-2.,0.3].iter().enumerate().map(|(i,&v)|tensor(&[v],&[1],DType::F32,flags&(1<<i)!=0)).collect::<Vec<_>>();
            let output=backward(&inputs[0],&inputs[1],&inputs[2]).unwrap();
            inputs[changed].detach().copy_from(&Tensor::from_vec(vec![9.],&[1]).unwrap()).unwrap();
            let selected=inputs.iter().filter(|v|v.requires_grad()).collect::<Vec<_>>();
            assert!(output[0].vjp_wrt_optional(&selected,&Tensor::ones(&[1]),false).is_err());
            assert!(output[1].vjp_wrt_optional(&selected,&Tensor::ones(&[1]),true).is_err());
            assert!(inputs.iter().all(|v|v.grad().is_none()));
        } }
        let g=tensor(&[0.7],&[1],DType::F32,true);let x=tensor(&[-2.],&[1],DType::F32,true);let w=tensor(&[0.3],&[1],DType::F32,true);
        let output=backward(&g,&x,&w).unwrap();
        for value in &output {value.detach().copy_from(&Tensor::from_vec(vec![17.],&[1]).unwrap()).unwrap();}
        let seed=Tensor::ones(&[1]);let result=Tensor::vjp_many_wrt_optional(&[&output[0],&output[1]],&[&g],&[&seed,&seed],false).unwrap();
        check(result[0].as_ref().unwrap(),&[-1.7]);
        let shared=tensor(&[-2.],&[1],DType::F32,true);let output=backward(&shared,&shared,&shared).unwrap();
        check(&output[0].add(&output[1]).unwrap().sum().grad_wrt(&[&shared],false).unwrap()[0],&[-8.]);
    }

    #[test]
    fn genuine_jvp_keeps_mixed_direction_types_and_both_graph_outputs() {
        use crate::forward_ad::DualLevel;
        let g=tensor(&[0.7,0.9],&[2],DType::F32,true);let x=tensor(&[-2.,0.],&[2],DType::F32,true);let w=tensor(&[0.3,0.4],&[2],DType::F32,true);
        let direction=tensor(&[2.,3.],&[2],DType::F64,true);
        let level=DualLevel::functional(&[&x],&[&direction]).unwrap();
        let output=backward(&g,&x,&w).unwrap();
        assert!(level.tangent(&output[0]).unwrap().is_none());
        let second=level.tangent(&output[1]).unwrap().unwrap();drop(level);
        assert_eq!(second.dtype(),DType::F64);check(&second,&[1.4,0.]);
        let result=second.sum().grad_wrt(&[&g,&direction],false).unwrap();
        check(&result[0],&[2.,0.]);check(&result[1],&[0.7,0.]);
    }

    #[test]
    fn direct_dual_preserves_genuine_missing_output_tangents() {
        use crate::forward_ad::DualLevel;
        // Actual Torch direct-dual controls distinguish None from func.jvp's
        // final ordinary zero materialization, including no-grad execution.
        for mask in 0..8 { for dtype in [DType::F32,DType::F64] { for live in [false,true] { for enabled in [false,true] {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,live),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,live),tensor(&[f64::INFINITY;7],&[7],DType::F32,live)];
            let directions=[0.2,0.3,0.4].iter().map(|&value|tensor(&[value;7],&[7],dtype,live)).collect::<Vec<_>>();
            let mut primals=inputs.iter().enumerate().filter_map(|(i,value)|if mask&(1<<i)!=0 {Some(value)}else{None}).collect::<Vec<_>>();
            let mut seeds=directions.iter().enumerate().filter_map(|(i,value)|if mask&(1<<i)!=0 {Some(value)}else{None}).collect::<Vec<_>>();
            let unrelated=tensor(&[0.;7],&[7],DType::F32,false);
            if mask==0 { primals.push(&unrelated);seeds.push(&directions[0]); }
            crate::autograd::with_grad_enabled(enabled,|| {
            let level=DualLevel::new(&primals,&seeds).unwrap();
            let pair=backward(&inputs[0],&inputs[1],&inputs[2]).unwrap();
            for (slot,present) in [mask&5!=0,mask&3!=0].into_iter().enumerate() {
                let tangent=level.tangent(&pair[slot]).unwrap();
                assert_eq!(tangent.is_some(),present,"mask {mask}, slot {slot}, dtype {dtype:?}, live {live}, enabled {enabled}");
                if let Some(value)=tangent {
                    assert_eq!(value.dtype(),dtype);
                    assert_eq!(value.requires_grad(),live&&enabled);
                    assert!(!value.is_structural_zero());
                }
            }
            });
        } } } }
    }

    #[test]
    fn all_missing_direction_masks_match_actual_torch_pair_jvp_nonfinite_controls() {
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 1&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F32,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F32);check(&outputs[0],&[f64::INFINITY,f64::INFINITY,0.20000000298023224,0.20000000298023224,0.20000000298023224,0.20000000298023224,f64::INFINITY]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F32);check(&outputs[1],&[f64::NEG_INFINITY,-0.4000000059604645,0.0,0.0,0.0,0.0,f64::NAN]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 1&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F64,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F64);check(&outputs[0],&[f64::INFINITY,f64::INFINITY,0.2,0.2,0.2,0.2,f64::INFINITY]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F64);check(&outputs[1],&[f64::NEG_INFINITY,-0.4,0.0,0.0,0.0,0.0,f64::NAN]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 2&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F32,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F32);check(&outputs[0],&[0.0,0.0,0.0,0.0,0.0,0.0,0.0]);
            assert_eq!(outputs[0].requires_grad(),false);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F32);check(&outputs[1],&[0.21000000834465027,0.21000000834465027,0.0,0.0,0.0,0.0,0.21000000834465027]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 2&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F64,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F32);check(&outputs[0],&[0.0,0.0,0.0,0.0,0.0,0.0,0.0]);
            assert_eq!(outputs[0].requires_grad(),false);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F64);check(&outputs[1],&[0.2099999964237213,0.2099999964237213,0.0,0.0,0.0,0.0,0.2099999964237213]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 3&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F32,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F32);check(&outputs[0],&[f64::INFINITY,f64::INFINITY,0.20000000298023224,0.20000000298023224,0.20000000298023224,0.20000000298023224,f64::INFINITY]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F32);check(&outputs[1],&[f64::NEG_INFINITY,-0.1899999976158142,0.0,0.0,0.0,0.0,f64::NAN]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 3&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F64,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F64);check(&outputs[0],&[f64::INFINITY,f64::INFINITY,0.2,0.2,0.2,0.2,f64::INFINITY]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F64);check(&outputs[1],&[f64::NEG_INFINITY,-0.19000000357627872,0.0,0.0,0.0,0.0,f64::NAN]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 4&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F32,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F32);check(&outputs[0],&[0.2800000011920929,0.2800000011920929,0.0,0.0,0.0,0.0,0.2800000011920929]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F32);check(&outputs[1],&[0.0,0.0,0.0,0.0,0.0,0.0,0.0]);
            assert_eq!(outputs[1].requires_grad(),false);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 4&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F64,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F64);check(&outputs[0],&[0.27999999523162844,0.27999999523162844,0.0,0.0,0.0,0.0,0.27999999523162844]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F32);check(&outputs[1],&[0.0,0.0,0.0,0.0,0.0,0.0,0.0]);
            assert_eq!(outputs[1].requires_grad(),false);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 5&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F32,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F32);check(&outputs[0],&[f64::INFINITY,f64::INFINITY,0.20000000298023224,0.20000000298023224,0.20000000298023224,0.20000000298023224,f64::INFINITY]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F32);check(&outputs[1],&[f64::NEG_INFINITY,-0.4000000059604645,0.0,0.0,0.0,0.0,f64::NAN]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 5&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F64,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F64);check(&outputs[0],&[f64::INFINITY,f64::INFINITY,0.2,0.2,0.2,0.2,f64::INFINITY]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F64);check(&outputs[1],&[f64::NEG_INFINITY,-0.4,0.0,0.0,0.0,0.0,f64::NAN]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 6&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F32,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F32);check(&outputs[0],&[0.2800000011920929,0.2800000011920929,0.0,0.0,0.0,0.0,0.2800000011920929]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F32);check(&outputs[1],&[0.21000000834465027,0.21000000834465027,0.0,0.0,0.0,0.0,0.21000000834465027]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 6&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F64,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F64);check(&outputs[0],&[0.27999999523162844,0.27999999523162844,0.0,0.0,0.0,0.0,0.27999999523162844]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F64);check(&outputs[1],&[0.2099999964237213,0.2099999964237213,0.0,0.0,0.0,0.0,0.2099999964237213]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 7&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F32,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F32);check(&outputs[0],&[f64::INFINITY,f64::INFINITY,0.20000000298023224,0.20000000298023224,0.20000000298023224,0.20000000298023224,f64::INFINITY]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F32);check(&outputs[1],&[f64::NEG_INFINITY,-0.1899999976158142,0.0,0.0,0.0,0.0,f64::NAN]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
        {
            let inputs=[tensor(&[0.7;7],&[7],DType::F32,true),tensor(&[f64::NEG_INFINITY,-2.,-0.,0.,2.,f64::INFINITY,f64::NAN],&[7],DType::F32,true),tensor(&[f64::INFINITY;7],&[7],DType::F32,true)];
            let directions=[0.2,0.3,0.4].iter().enumerate().map(|(index,&value)|if 7&(1<<index)!=0 {Some(tensor(&[value;7],&[7],DType::F64,true))}else{None}).collect::<Vec<_>>();
            let outputs=jvp(&inputs,&directions).unwrap().into_iter().map(|value|value.unwrap_or_else(||zeros_like(&inputs[0]).unwrap())).collect::<Vec<_>>();
            assert_eq!(outputs[0].dtype(),DType::F64);check(&outputs[0],&[f64::INFINITY,f64::INFINITY,0.2,0.2,0.2,0.2,f64::INFINITY]);
            assert_eq!(outputs[0].requires_grad(),true);assert_eq!(outputs[0].is_structural_zero(),false);
            assert_eq!(outputs[1].dtype(),DType::F64);check(&outputs[1],&[f64::NEG_INFINITY,-0.19000000357627872,0.0,0.0,0.0,0.0,f64::NAN]);
            assert_eq!(outputs[1].requires_grad(),true);assert_eq!(outputs[1].is_structural_zero(),false);
        }
    }

    #[test]
    fn empty_paths_and_dtype_shape_capacity_errors_are_bounded() {
        let empty=Tensor::zeros(&[1usize<<45,0]);let output=backward(&empty,&empty,&empty).unwrap();
        assert!(output.iter().all(|v|v.numel()==0));
        let gradients=graph_vjp(&[empty.clone(),empty.clone(),empty.clone()],&[Some(empty.clone()),None]).unwrap();
        assert!(gradients.iter().all(|v|v.as_ref().unwrap().numel()==0));
        assert!(jvp(&[empty.clone(),empty.clone(),empty],&[None,None,None]).unwrap().iter().all(Option::is_none));
        let enormous=Tensor::scalar(1.).expand(&[isize::MAX as usize/8+1]).unwrap();
        assert!(backward(&enormous,&enormous,&enormous).is_err());
        assert!(backward(&Tensor::ones(&[2]),&Tensor::ones(&[3]),&Tensor::ones(&[3])).is_err());
        assert!(backward(&Tensor::from_vec_i64(vec![1],&[]).unwrap(),&Tensor::scalar(1.),&Tensor::scalar(1.)).is_err());
    }
}


pub(crate) fn jvp_physical(inputs: &[Tensor], tangents: &[Option<Tensor>], batch: usize) -> Result<Vec<Option<Tensor>>> {
    validate(inputs)?;
    if tangents.len() != 3 { return Err(Error::InvalidShape { op: OP, msg: "expected three input tangent slots".into() }); }
    let mut shape = vec![batch]; shape.extend_from_slice(inputs[1].shape());
    crate::typed_math::check_shape(OP, &shape)?;
    for direction in tangents.iter().flatten() {
        if !matches!(direction.dtype(), DType::F32 | DType::F64) { return Err(Error::DtypeMismatch { op: OP, expected: DType::F32, got: direction.dtype() }); }
        crate::typed_math::check(OP, &[direction], direction.dtype())?;
        if direction.shape() != shape { return Err(Error::ShapeMismatch { op: OP, lhs: shape.clone(), rhs: direction.shape().to_vec() }); }
    }
    if tangents.iter().all(Option::is_none) { return Ok(vec![None, None]); }
    let directions = inputs.iter().zip(tangents).map(|(input, direction)| match direction {
        Some(value) => Ok(value.clone()),
        None => Tensor::structural_zero(&[], input.dtype(), Device::Cpu)?.expand(input.shape()),
    }).collect::<Result<Vec<_>>>()?;
    let (g, x, w) = (&inputs[0], &inputs[1], &inputs[2]);
    let (dg, dx, dw) = (&directions[0], &directions[1], &directions[2]);
    let positive = x.ge_tensor(&zero(x.dtype())?)?;
    // Where plans the actual condition-first physical layout. Keep the
    // arithmetic on logical operands, including missing-direction adjoints.
    let first = if tangents[0].is_some() || tangents[2].is_some() {
        let negative = dg.mul(w)?.add(&g.mul(dw)?)?;
        Some(select(&positive, dg, &negative)?)
    } else { None };
    let second = if tangents[0].is_some() || tangents[1].is_some() {
        let negative = g.mul(dx)?.add(&dg.mul(x)?)?;
        Some(select(&positive, &zero(negative.dtype())?, &negative)?)
    } else { None };
    Ok(vec![first, second])
}

#[cfg(test)]
#[path = "../../tests/prelu_pair_physical/mod.rs"]
mod physical_tests;
