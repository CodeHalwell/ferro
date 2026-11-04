//! Saved-output activation derivatives, recorded on the existing Op graph.
use crate::{DType, Device, Error, Result, Tensor};
use crate::autograd::{ForwardOp, with_grad_enabled};
use crate::dispatch::BinaryKind;
use crate::tensor::raw_binary_k;

#[derive(Clone, Copy)]
pub(crate) enum Kind { Sigmoid, Tanh }

fn constant(reference: &Tensor, value: f32) -> Result<Tensor> {
    if reference.dtype() == DType::F64 { crate::typed_math::full_f64(&[], value as f64) }
    else { Tensor::full_on(&[], value, reference.device()) }
}

pub(crate) fn backward(kind: Kind, gradient: &Tensor, output: &Tensor) -> Result<Tensor> {
    let op = match kind { Kind::Sigmoid => "sigmoid_backward", Kind::Tanh => "tanh_backward" };
    if gradient.shape() != output.shape() { return Err(Error::ShapeMismatch { op, lhs: gradient.shape().to_vec(), rhs: output.shape().to_vec() }); }
    if gradient.device() != output.device() { return Err(Error::DeviceMismatch { op, lhs: gradient.device(), rhs: output.device() }); }
    for input in [gradient, output] {
        if !matches!(input.dtype(), DType::F32 | DType::F64) { return Err(Error::DtypeMismatch { op, expected: DType::F32, got: input.dtype() }); }
    }
    let raw = if gradient.device() == Device::Cpu {
        crate::typed_math::pointwise_binary_real(op, gradient, output,
            |g, y| match kind { Kind::Sigmoid => (g * (1.0 - y)) * y, Kind::Tanh => g * (-y).mul_add(y, 1.0) },
            |g, y| match kind { Kind::Sigmoid => (g * (1.0 - y)) * y, Kind::Tanh => g * (-y).mul_add(y, 1.0) })?
    } else {
        for input in [gradient, output] {
            if input.dtype() != DType::F32 { return Err(Error::DtypeMismatch { op, expected: DType::F32, got: input.dtype() }); }
        }
        // Retain the existing F32 resident primitives; device rounding is a
        // separate hardware acceptance boundary from the CPU fused kernel.
        let materialize = |input: &Tensor| -> Result<Tensor> {
            if input.is_structural_zero() { with_grad_enabled(false, || input.clone_tensor(false)).map(|value| value.detach()) }
            else { Ok(input.clone()) }
        };
        let (g, y) = (materialize(gradient)?, materialize(output)?);
        let one = Tensor::full_on(y.shape(), 1.0, y.device())?;
        match kind {
            Kind::Sigmoid => {
                let difference = raw_binary_k(op, &one, &y, BinaryKind::Sub)?;
                let first = raw_binary_k(op, &g, &difference, BinaryKind::Mul)?;
                raw_binary_k(op, &first, &y, BinaryKind::Mul)?
            }
            Kind::Tanh => {
                let square = raw_binary_k(op, &y, &y, BinaryKind::Mul)?;
                let difference = raw_binary_k(op, &one, &square, BinaryKind::Sub)?;
                raw_binary_k(op, &g, &difference, BinaryKind::Mul)?
            }
        }
    };
    let forward = match kind { Kind::Sigmoid => ForwardOp::SigmoidBackward, Kind::Tanh => ForwardOp::TanhBackward };
    if !gradient.requires_grad() && !output.requires_grad() && !crate::forward_ad::is_active() {
        return Ok(if crate::capture::is_recording() { raw.record_fn_forward(vec![gradient.clone(), output.clone()], forward, |_| unreachable!("inference derivative has no backward")) } else { raw });
    }
    // These are parent operands, never this node's own output. Their saved
    // versions guard writes while retaining exact broadcast/strided layouts.
    let (g, y) = (gradient.clone(), output.clone());
    Ok(raw.record_fn_forward(vec![gradient.clone(), output.clone()], forward, move |incoming| {
        vjp(kind, incoming, &g, &y).expect("validated activation derivative operands")
    }))
}

pub(crate) fn vjp(kind: Kind, incoming: &Tensor, gradient: &Tensor, output: &Tensor) -> Result<Vec<Tensor>> {
    let dg = backward(kind, incoming, output)?;
    let dy = match kind {
        Kind::Sigmoid => {
            let factor = output.mul(&constant(output, -2.0)?)?.add(&constant(output, 1.0)?)?;
            incoming.mul(gradient)?.mul(&factor)?
        }
        Kind::Tanh => incoming.mul(&output.mul(&constant(output, -2.0)?)?.mul(gradient)?)?,
    };
    Ok(vec![dg.to_dtype_autograd(gradient.dtype(), false)?, dy.to_dtype_autograd(output.dtype(), false)?])
}

pub(crate) fn jvp(kind: Kind, gradient: &Tensor, output: &Tensor, dg: &Tensor, dy: &Tensor) -> Result<Tensor> {
    let first = backward(kind, dg, output)?;
    let second = match kind {
        Kind::Sigmoid => {
            let factor = output.mul(&constant(output, -2.0)?)?.add(&constant(output, 1.0)?)?;
            dy.mul(gradient)?.mul(&factor)?
        }
        Kind::Tanh => dy.mul(&output.mul(&constant(output, -2.0)?)?.mul(gradient)?)?,
    };
    first.add(&second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_ad::DualLevel;
    fn leaf(values: &[f32]) -> Tensor { Tensor::from_vec(values.to_vec(), &[values.len()]).unwrap().requires_grad_(true).unwrap() }
    fn exact(a: f32, b: f32) { assert!(a.is_nan() && b.is_nan() || a.to_bits() == b.to_bits(), "{a:?} != {b:?}"); }

    #[test]
    fn backward_callbacks_preserve_parent_layout_and_release_all_ancestor_handles() {
        for kind in [Kind::Sigmoid,Kind::Tanh] {
            let gradient=Tensor::full(&[2,3],0.7).transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
            let value=Tensor::full(&[2,3],0.2).transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
            let weak_gradient=std::sync::Arc::downgrade(&gradient.0);
            let weak_value=std::sync::Arc::downgrade(&value.0);
            let output=backward(kind,&gradient,&value).unwrap();
            let weak_output=std::sync::Arc::downgrade(&output.0);
            let seed=Tensor::scalar(1.).expand(output.shape()).unwrap();
            let result=Tensor::vjp_many_wrt_optional(&[&output],&[&gradient,&value],&[&seed],false).unwrap();
            assert_eq!(result[0].as_ref().unwrap().strides(),&[1,3]);
            assert_eq!(result[1].as_ref().unwrap().strides(),&[1,3]);
            for parent in [&gradient,&value] {
                parent.transpose(0,1).unwrap().detach().copy_from(&Tensor::zeros(&[2,3])).unwrap();
                assert!(output.vjp_wrt(&[&gradient,&value],&seed,false).is_err());
            }
            assert!(gradient.grad().is_none()&&value.grad().is_none());
            drop(output);drop(gradient);drop(value);
            assert!(weak_output.upgrade().is_none());
            assert!(weak_gradient.upgrade().is_none());assert!(weak_value.upgrade().is_none());
        }
    }

    #[test]
    fn ordinary_backward_through_live_derivatives_releases_graphs_and_publishes_detached_slots() {
        for tanh in [false,true] {for transpose in [false,true] {
            let values=Tensor::from_vec(vec![0.2,0.4,0.3,0.7,0.1,0.6],&[2,3]).unwrap();
            let x=if transpose{values.transpose(0,1).unwrap()}else{values};
            let x=x.detach().requires_grad_(true).unwrap();
            let y=if tanh{x.tanh()}else{x.sigmoid()};
            let first=y.sum().grad_wrt(&[&x],true).unwrap().remove(0);
            let weak_x=std::sync::Arc::downgrade(&x.0);let weak_y=std::sync::Arc::downgrade(&y.0);let weak_first=std::sync::Arc::downgrade(&first.0);
            first.sum().backward();
            let gradient=x.grad().unwrap();assert!(!gradient.requires_grad());assert!(gradient.is_leaf());assert_eq!(gradient.strides(),x.strides());
            assert!(crate::autograd::is_grad_enabled());
            drop(first);drop(y);drop(x);
            assert!(weak_x.upgrade().is_none()&&weak_y.upgrade().is_none()&&weak_first.upgrade().is_none());
        }}
    }

    #[test]
    fn backward_primitives_keep_both_saved_inputs_but_not_the_output() {
        for kind in [Kind::Sigmoid, Kind::Tanh] { for live_g in [false,true] { for live_y in [false,true] {
            if !live_g && !live_y { continue; }
            for mutate in 0..3 {
                let g = leaf(&[0.7,-0.3]).requires_grad_(live_g).unwrap();
                let y = leaf(&[0.2,-0.7]).requires_grad_(live_y).unwrap();
                let out = backward(kind,&g,&y).unwrap();
                let selected = if live_g { &g } else { &y };
                let old = out.sum().grad_wrt(&[selected],false).unwrap().remove(0);
                let target = [&g,&y,&out][mutate];
                target.detach().copy_from(&Tensor::ones(&[2])).unwrap();
                for graph in [false,true] {
                    let result = out.sum().grad_wrt(&[selected],graph);
                    assert_eq!(result.is_ok(),mutate==2);
                    if mutate==2 { assert_eq!(result.unwrap()[0].to_vec(),old.to_vec()); }
                }
                assert!(g.grad().is_none() && y.grad().is_none());
            }
        } } }
    }

    #[test]
    fn backward_primitive_vjp_and_jvp_stages_keep_missing_nonfinite_terms() {
        for kind in [Kind::Sigmoid, Kind::Tanh] {
            let g = leaf(&[0.7,2.,2.,2.,0.,0.]);
            let y = leaf(&[0.2,f32::INFINITY,f32::NEG_INFINITY,f32::NAN,f32::INFINITY,f32::NEG_INFINITY]);
            let incoming = leaf(&[0.3,1.,1.,1.,1.,1.]);
            let out = backward(kind,&g,&y).unwrap();
            for graph in [false,true] {
                let gradients = out.vjp_wrt(&[&g,&y],&incoming,graph).unwrap();
                for (i,((actual_g,actual_y),seed)) in gradients[0].to_vec().into_iter().zip(gradients[1].to_vec()).zip(incoming.to_vec()).enumerate() {
                    let (gv,yv)=(g.to_vec()[i],y.to_vec()[i]);
                    let expected_g=match kind { Kind::Sigmoid=>(seed*(1.-yv))*yv,Kind::Tanh=>seed*(-yv).mul_add(yv,1.) };
                    let expected_y=match kind { Kind::Sigmoid=>(seed*gv)*(1.-2.*yv),Kind::Tanh=>seed*((-2.*yv)*gv) };
                    exact(actual_g,expected_g); exact(actual_y,expected_y);
                }
            }
            for only_g in [false,true] {
                let source=if only_g {&g} else {&y};
                let level=DualLevel::new(&[source],&[&incoming]).unwrap();
                let out=backward(kind,&g,&y).unwrap();
                let tangent=level.tangent(&out).unwrap().unwrap();
                drop(level);
                assert!(tangent.requires_grad());
                for actual in &tangent.to_vec()[1..] {
                    if only_g { assert!(actual.is_infinite() && actual.is_sign_negative() || actual.is_nan()); }
                    else { assert!(actual.is_nan()); }
                }
                if only_g {
                    let derivative=tangent.sum().grad_wrt(&[&g],true).unwrap().remove(0);
                    assert!(derivative.is_structural_zero() && derivative.requires_grad());
                }
            }
        }
    }

    #[test]
    fn backward_primitive_mixed_direction_stages_promote_only_at_each_operation() {
        for kind in [Kind::Sigmoid,Kind::Tanh] { for only_g in [false,true] {
            let g=leaf(&[0.7]); let y=leaf(&[0.2]);
            let direction=Tensor::from_vec_f64(vec![0.3],&[1]).unwrap().requires_grad_(true).unwrap();
            let source=if only_g {&g} else {&y};
            let level=DualLevel::new(&[source],&[&direction]).unwrap();
            let out=backward(kind,&g,&y).unwrap();
            let tangent=level.tangent(&out).unwrap().unwrap();
            drop(level);
            let (gv,yv)=(0.7f32,0.2f32);
            let expected=if only_g { match kind { Kind::Sigmoid=>(0.3*(1.-yv as f64))*yv as f64,Kind::Tanh=>0.3*(-(yv as f64)).mul_add(yv as f64,1.) } }
                else { match kind { Kind::Sigmoid=>(0.3*gv as f64)*(1.-2.*yv) as f64,Kind::Tanh=>0.3*((-2.*yv)*gv) as f64 } };
            assert_eq!(tangent.dtype(),DType::F64);
            assert_eq!(tangent.to_vec_f64()[0].to_bits(),expected.to_bits());
            assert!(tangent.requires_grad());
        } }
    }
}
