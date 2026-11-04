use ferro_core::{Tensor, DType};
use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use std::sync::{Arc, Mutex};

fn leaf(value: f32) -> Tensor { Tensor::scalar(value).requires_grad_(true).unwrap() }

#[test]
fn recording_rejects_shared_or_nonfresh_outputs_before_attaching_any_edge() {
    let x = leaf(2.0);
    let alias = Tensor::scalar(3.0);
    assert!(Tensor::record_many_optional(vec![alias.clone(), alias.clone()], vec![x.clone()], &[true,true], |_| vec![None]).is_err());
    assert!(!alias.requires_grad());
    assert!(Tensor::record_many_optional(vec![leaf(1.)], vec![x.clone()], &[false], |_| vec![None]).is_err());
    let interior = x.square().unwrap();
    assert!(Tensor::record_many_optional(vec![interior], vec![x.clone()], &[true], |_| vec![None]).is_err());
    assert!(Tensor::record_many_optional(vec![], vec![x.clone()], &[], |_| vec![None]).is_err());
    assert!(Tensor::record_many_optional(vec![Tensor::scalar(0.)], vec![x.clone()], &[true,false], |_| vec![None]).is_err());
    assert!(Tensor::record_many_optional(vec![Tensor::from_vec_i64(vec![0], &[]).unwrap()], vec![x], &[true], |_| vec![None]).is_err());
}

#[test]
fn typed_differentiable_slots_and_empty_output_shapes_preserve_real_optional_inputs() {
    let x = Tensor::from_vec_f64(vec![2.0], &[]).unwrap().requires_grad_(true).unwrap();
    let out = Tensor::record_many_optional(vec![Tensor::from_vec_f64(vec![3.0], &[]).unwrap(), Tensor::from_vec_i64(vec![7], &[]).unwrap()], vec![x.clone()], &[true,false], |seeds| {
        assert!(seeds[1].is_none()); vec![seeds[0].clone()]
    }).unwrap();
    assert!(out[0].requires_grad()); assert!(!out[1].requires_grad());
    out[0].backward(); assert_eq!(x.grad().unwrap().to_vec_f64(), [1.]);
    let x = leaf(1.);
    let out = Tensor::record_many_optional(vec![Tensor::zeros(&[0,usize::MAX]), Tensor::scalar(0.)], vec![x.clone()], &[true,true], |seeds| {
        assert_eq!(seeds[0].as_ref().unwrap().shape(), [0,usize::MAX]);
        assert!(seeds[1].is_none()); vec![None]
    }).unwrap();
    out[0].backward_with(&Tensor::zeros(&[0,usize::MAX]));
    assert!(x.grad().is_none()); assert!(out[1].grad().is_none());
}

#[test]
fn foundation_boundaries_restore_modes_and_do_not_invoke_callbacks() {
    let x=leaf(2.);
    with_grad_enabled(false, || {
        let out=Tensor::record_many_optional(vec![Tensor::scalar(1.),Tensor::scalar(2.)],vec![x.clone()],&[true,true], |_| panic!("unrecorded callback")).unwrap();
        assert!(out.iter().all(|value| !value.requires_grad()));
    });
    let out=Tensor::record_many_optional(vec![Tensor::scalar(1.),Tensor::scalar(2.)],vec![x.clone()],&[true,true], |_| panic!("unsupported callback must not run")).unwrap();
    assert!(out[0].grad_wrt(&[&x],true).is_err()); assert!(is_grad_enabled());
    let direction=Tensor::scalar(1.);
    let level=ferro_core::forward_ad::DualLevel::new(&[&x],&[&direction]).unwrap();
    assert!(Tensor::record_many_optional(vec![Tensor::scalar(1.),Tensor::scalar(2.)],vec![x.clone()],&[true,true], |_| vec![None]).is_err());
    drop(level);
    assert!(ferro_core::capture::capture(|| Tensor::record_many_optional(vec![Tensor::scalar(1.),Tensor::scalar(2.)],vec![x.clone()],&[true,true], |_| vec![None]).map(|mut outputs| outputs.remove(0))).is_err());
    assert!(is_grad_enabled());
    assert_eq!(x.square().unwrap().grad_wrt(&[&x],false).unwrap()[0].dtype(), DType::F32);
}

#[test]
fn legacy_callback_order_and_live_selected_seed_identity_remain_unchanged() {
    let order=Arc::new(Mutex::new(Vec::new()));
    let x=leaf(1.);
    let record=|input: &Tensor, label| {
        let order=order.clone();
        Tensor::scalar(1.).record_fn(vec![input.clone()],move |g| { order.lock().unwrap().push(label); vec![g.clone()] })
    };
    let a=record(&x,0);let b=record(&a,1);let c=record(&a,2);let total=b.add(&c).unwrap();
    total.backward(); assert_eq!(*order.lock().unwrap(), [2,1,0]);
    order.lock().unwrap().clear();
    assert_eq!(total.grad_wrt(&[&x],false).unwrap()[0].item(),2.);
    assert_eq!(*order.lock().unwrap(),[2,1,0]);
    let seed=leaf(3.);
    let result=Tensor::vjp_many_wrt_optional(&[&b], &[&b], &[&seed], false).unwrap();
    assert_eq!(result[0].as_ref().unwrap().id(),seed.id());
    assert_eq!(*order.lock().unwrap(),[2,1,0]);
    let input=leaf(1.);
    let outputs=Tensor::record_many_optional(vec![Tensor::scalar(1.),Tensor::scalar(2.)],vec![input.clone()],&[true,true],|seeds| {
        vec![seeds[0].clone().or_else(|| seeds[1].clone())]
    }).unwrap();
    outputs[1].backward();
    let held=outputs[1].grad().unwrap();
    outputs[1].detach().copy_from(&Tensor::scalar(9.)).unwrap();
    outputs[0].backward();
    assert_eq!(outputs[1].grad().unwrap().id(),held.id());
    assert_eq!(held.item(),1.);
    assert_eq!(input.grad().unwrap().item(),2.);
}
