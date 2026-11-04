use ferro_core::{Tensor, DType};
fn leaf(v: f32) -> Tensor { Tensor::scalar(v).requires_grad_(true).unwrap() }

#[test]
fn duplicate_roots_sum_seeds_before_visiting_a_shared_operation() {
    let x = leaf(1e20); let out = x.square().unwrap();
    let positive = leaf(1e20); let negative = leaf(-1e20);
    for graph in [false, true] {
        let result = Tensor::vjp_many_wrt_optional(&[&out, &out], &[&x], &[&positive, &negative], graph).unwrap().remove(0).unwrap();
        assert_eq!(result.item(), 0.);
        assert_eq!(result.requires_grad(), graph);
        assert!(out.vjp_wrt(&[&x], &positive, false).unwrap()[0].add(&out.vjp_wrt(&[&x], &negative, false).unwrap()[0]).unwrap().item().is_nan());
    }
    assert!(x.grad().is_none() && positive.grad().is_none());
}

#[test]
fn ancestor_roots_duplicate_inputs_and_passthrough_seed_identity() {
    let x = leaf(2.); let out = x.square().unwrap(); let left = leaf(3.); let right = leaf(5.);
    for graph in [false, true] {
        let result = Tensor::vjp_many_wrt_optional(&[&x, &out], &[&x, &out, &x], &[&left, &right], graph).unwrap();
        assert_eq!(result[0].as_ref().unwrap().item(), 23.);
        assert_eq!(result[1].as_ref().unwrap().id(), right.id());
        assert_eq!(result[0].as_ref().unwrap().id(), result[2].as_ref().unwrap().id());
        assert_eq!(result[0].as_ref().unwrap().requires_grad(), graph);
    }
    let result = Tensor::vjp_many_wrt_optional(&[&x], &[&x], &[&left], false).unwrap();
    assert_eq!(result[0].as_ref().unwrap().id(), left.id());
    assert!(!x.vjp_wrt(&[&x], &left, false).unwrap()[0].requires_grad());
}

#[test]
fn mixed_outputs_preserve_original_input_types_and_higher_seed_derivatives() {
    let x = leaf(2.); let y = Tensor::from_vec_f64(vec![3.], &[]).unwrap().requires_grad_(true).unwrap();
    let out = x.add(&y).unwrap(); let a = Tensor::scalar(1.); let b = Tensor::from_vec_f64(vec![2.], &[]).unwrap();
    let result = Tensor::vjp_many_wrt_optional(&[&x, &out], &[&x, &y], &[&a, &b], false).unwrap();
    assert_eq!(result[0].as_ref().unwrap().dtype(), DType::F32); assert_eq!(result[0].as_ref().unwrap().item(), 3.);
    assert_eq!(result[1].as_ref().unwrap().to_vec_f64(), vec![2.]);
    let out = x.square().unwrap(); let first = leaf(0.3); let second = leaf(0.4);
    let gradient = Tensor::vjp_many_wrt_optional(&[&out, &out], &[&x], &[&first, &second], true).unwrap().remove(0).unwrap();
    let next = gradient.grad_wrt(&[&first, &second, &x], false).unwrap();
    assert_eq!(next[0].item(), 4.); assert_eq!(next[1].item(), 4.); assert!((next[2].item()-1.4).abs()<1e-6);
}

#[test]
fn absence_empty_roots_and_connected_zero_are_distinct_without_touching_slots() {
    let x=leaf(2.);let unused=leaf(7.);let zero=x.mul(&Tensor::scalar(0.)).unwrap();
    let result=Tensor::vjp_many_wrt_optional(&[&zero],&[&x,&unused],&[&Tensor::scalar(1.)],false).unwrap();
    assert_eq!(result[0].as_ref().unwrap().item(),0.);assert!(result[1].is_none());
    assert!(Tensor::vjp_many_wrt_optional(&[],&[&x],&[],false).unwrap()[0].is_none());
    assert_eq!(zero.grad_wrt(&[&unused],false).unwrap()[0].item(),0.);
    assert!(x.grad().is_none()&&unused.grad().is_none());
}

#[test]
fn malformed_late_seed_or_stale_root_never_changes_leaf_slots() {
    let x=leaf(2.);let y=leaf(3.);x.sum().backward();y.sum().backward();
    let before=(x.grad().unwrap().id(),y.grad().unwrap().id());
    let a=x.square().unwrap();let b=y.square().unwrap();
    assert!(Tensor::vjp_many_wrt_optional(&[&a,&b],&[&x,&y],&[&Tensor::scalar(1.),&Tensor::ones(&[2])],false).is_err());
    y.detach().copy_from(&Tensor::scalar(4.)).unwrap();
    assert!(Tensor::vjp_many_wrt_optional(&[&a,&b],&[&x,&y],&[&Tensor::scalar(1.),&Tensor::scalar(1.)],false).is_err());
    assert_eq!((x.grad().unwrap().id(),y.grad().unwrap().id()),before);
    assert_eq!(x.grad().unwrap().item(),1.);assert_eq!(y.grad().unwrap().item(),1.);
}

#[test]
fn joint_target_masks_keep_unsigned_alpha_left_derivative_available() {
    let x=leaf(2.);let y=leaf(3.);
    let a=x.add_alpha(&y,ferro_core::ops_ext::add_alpha::Alpha::U64(u64::MAX)).unwrap();
    let b=x.square().unwrap();let seed=Tensor::scalar(1.);
    let result=Tensor::vjp_many_wrt_optional(&[&a,&b],&[&x],&[&seed,&seed],true).unwrap();
    assert_eq!(result[0].as_ref().unwrap().item(),5.);
    assert!(Tensor::vjp_many_wrt_optional(&[&a,&b],&[&x,&y],&[&seed,&seed],true).is_err());
}
