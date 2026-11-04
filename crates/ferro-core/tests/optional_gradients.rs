use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use ferro_core::Tensor;

fn leaf(value: f32) -> Tensor { Tensor::scalar(value).requires_grad_(true).unwrap() }

#[test]
fn none_skips_upstream_nodes_and_functional_queries_return_zeros() {
    let x = leaf(2.0);
    let upstream = Tensor::scalar(7.0).record_fn(vec![x.clone()], |_| panic!("unused branch must not run"));
    let out = Tensor::scalar(3.0).record_fn_optional(vec![upstream.clone()], |_| vec![None]);
    out.backward();
    assert!(x.grad().is_none() && upstream.grad().is_none());
    assert_eq!(out.grad().unwrap().item(), 1.0);
    for create_graph in [false, true] {
        let derivatives = out.grad_wrt(&[&x, &upstream], create_graph).unwrap();
        assert!(derivatives.iter().all(|g| g.item() == 0.0 && !g.requires_grad()));
        assert!(x.grad().is_none() && upstream.grad().is_none());
    }
}

#[test]
fn partial_none_and_mixed_output_branches_accumulate_only_present_gradients() {
    let x = leaf(2.0);
    let y = leaf(3.0);
    let a = x.mul(&x).unwrap();
    let b = y.mul(&y).unwrap();
    let out = Tensor::scalar(9.0).record_fn_optional(vec![a.clone(), b.clone()], |g| vec![None, Some(g.clone())]);
    out.backward();
    assert!(a.grad().is_none() && x.grad().is_none());
    assert_eq!(y.grad().unwrap().item(), 6.0);
    let total = out.add(&x.mul(&x).unwrap()).unwrap();
    total.backward();
    assert_eq!(x.grad().unwrap().item(), 4.0);
    assert_eq!(y.grad().unwrap().item(), 12.0);
    assert!(a.grad().is_none());
    let derivatives = total.grad_wrt(&[&x, &y], false).unwrap();
    assert_eq!(derivatives.iter().map(Tensor::item).collect::<Vec<_>>(), vec![4.0, 6.0]);
    assert!(total.grad_wrt(&[&x, &y], true).is_err(), "custom present adjoints need an explicit higher-order rule");
}

#[test]
fn duplicate_inputs_and_changing_none_do_not_erase_existing_leaf_gradients() {
    let x = leaf(2.0);
    let mode = Arc::new(AtomicUsize::new(0));
    let captured = mode.clone();
    let out = Tensor::scalar(4.0).record_fn_optional(vec![x.clone(), x.clone(), x.clone()], move |g| {
        if captured.load(Ordering::Relaxed) == 0 { vec![Some(g.clone()), None, Some(g.clone())] }
        else { vec![None, None, None] }
    });
    out.backward();
    let held = x.grad().unwrap();
    assert_eq!(held.item(), 2.0);
    mode.store(1, Ordering::Relaxed);
    out.backward();
    assert_eq!(x.grad().unwrap().item(), 2.0);
    assert_eq!(held.item(), 2.0);
    assert_eq!(out.grad_wrt(&[&x], true).unwrap()[0].item(), 0.0);
}

#[test]
fn invalid_optional_callbacks_publish_no_gradient_slots() {
    for invalid in 0..3 {
        let x = leaf(2.0);
        let y = leaf(3.0);
        x.mul(&x).unwrap().backward();
        y.backward();
        let held_x = x.grad().unwrap();
        let held_y = y.grad().unwrap();
        let bad = Tensor::scalar(5.0).record_fn_optional(vec![x.clone(), y.clone()], move |g| match invalid {
            0 => vec![Some(g.clone())],
            1 => vec![Some(g.clone()), Some(Tensor::ones(&[2]))],
            _ => vec![Some(g.clone()), Some(Tensor::from_vec_i64(vec![1], &[]).unwrap())],
        });
        let good = x.mul(&x).unwrap();
        let total = bad.add(&good).unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| total.backward())).is_err());
        assert!(total.grad().is_none() && good.grad().is_none() && bad.grad().is_none());
        assert_eq!(x.grad().unwrap().id(), held_x.id());
        assert_eq!(y.grad().unwrap().id(), held_y.id());
        assert_eq!((held_x.item(), held_y.item()), (4.0, 1.0));
        assert!(total.grad_wrt(&[&x, &y], false).is_err());
        assert_eq!((x.grad().unwrap().item(), y.grad().unwrap().item()), (4.0, 1.0));
    }
}

#[test]
fn saved_versions_are_checked_even_for_none_inputs_before_any_publication() {
    let base = Tensor::scalar(2.0);
    let x = base.requires_grad_(true).unwrap();
    let y = leaf(3.0);
    y.mul(&y).unwrap().backward();
    let calls = Arc::new(AtomicUsize::new(0));
    let captured = calls.clone();
    let unused = Tensor::scalar(1.0).record_fn_optional(vec![x.clone()], move |_| {
        captured.fetch_add(1, Ordering::Relaxed);
        vec![None]
    });
    let good = y.mul(&y).unwrap();
    let total = unused.add(&good).unwrap();
    base.copy_from(&Tensor::scalar(7.0)).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| total.backward())).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(total.grad().is_none() && good.grad().is_none() && unused.grad().is_none() && x.grad().is_none());
    assert_eq!(y.grad().unwrap().item(), 6.0);
    for create_graph in [false, true] { assert!(total.grad_wrt(&[&x, &y], create_graph).is_err()); }
}

#[test]
fn float64_none_and_no_grad_registration_keep_existing_contracts() {
    let x = Tensor::from_vec_f64(vec![2.0], &[]).unwrap().requires_grad_(true).unwrap();
    let out = Tensor::from_vec_f64(vec![1.0], &[]).unwrap().record_fn_optional(vec![x.clone()], |_| vec![None]);
    out.backward();
    assert!(x.grad().is_none());
    assert_eq!(out.grad_wrt(&[&x], false).unwrap()[0].to_vec_f64(), vec![0.0]);
    assert!(out.grad_wrt(&[&x], true).is_err());
    with_grad_enabled(false, || {
        let detached = Tensor::scalar(3.0).record_fn_optional(vec![x.clone()], |_| vec![None]);
        assert!(!detached.requires_grad());
        assert!(!is_grad_enabled());
    });
    assert!(is_grad_enabled());
}

#[test]
fn strict_transform_seam_distinguishes_unused_from_numerical_zero() {
    let x = leaf(2.0);
    let absent = Tensor::scalar(4.0).record_fn_optional(vec![x.clone()], |_| vec![None]);
    let zero = x.mul(&Tensor::scalar(0.0)).unwrap();
    for create_graph in [false, true] {
        let none = absent.vjp_wrt_optional(&[&x], &Tensor::scalar(1.0), create_graph).unwrap();
        let present = zero.vjp_wrt_optional(&[&x], &Tensor::scalar(1.0), create_graph).unwrap();
        assert!(none[0].is_none());
        assert_eq!(present[0].as_ref().unwrap().item(), 0.0);
        assert_eq!(absent.grad_wrt(&[&x], create_graph).unwrap()[0].item(), 0.0);
        assert_eq!(zero.grad_wrt(&[&x], create_graph).unwrap()[0].item(), 0.0);
    }
    assert!(x.grad().is_none());
    let y = leaf(3.0);
    let out = x.add(&Tensor::scalar(1.0)).unwrap();
    let derivatives = out.vjp_wrt_optional(&[&x, &y], &Tensor::scalar(2.0), false).unwrap();
    assert_eq!(derivatives[0].as_ref().unwrap().item(), 2.0);
    assert!(derivatives[1].is_none());
    assert!(x.grad().is_none() && y.grad().is_none());
}
