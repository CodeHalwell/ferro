use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use ferro_core::{DType, Tensor};
use ferro_core::autograd::is_grad_enabled;

fn leaf(value: f32) -> Tensor { Tensor::scalar(value).requires_grad_(true).unwrap() }

#[test]
fn sibling_consumers_reach_one_callback_with_all_accumulated_cotangents() {
    let x = leaf(2.);
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(Mutex::new(Vec::new()));
    let count = calls.clone(); let seen = observed.clone();
    let outputs = Tensor::record_many_optional(vec![Tensor::scalar(4.), Tensor::scalar(6.)], vec![x.clone()], &[true, true], move |seeds| {
        assert!(!is_grad_enabled());
        count.fetch_add(1, Ordering::Relaxed);
        let values = seeds.iter().map(|g| g.as_ref().map(Tensor::item)).collect::<Vec<_>>();
        seen.lock().unwrap_or_else(|e| e.into_inner()).push(values.clone());
        vec![Some(Tensor::scalar(2. * values[0].unwrap_or(0.) + 3. * values[1].unwrap_or(0.)))]
    }).unwrap();
    let left = outputs[0].square().unwrap();
    let right = outputs[1].mul(&Tensor::scalar(5.)).unwrap();
    let total = left.add(&right).unwrap();
    assert_eq!(total.grad_wrt(&[&x], false).unwrap()[0].item(), 31.);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(x.grad().is_none() && outputs.iter().all(|value| value.grad().is_none()));
    total.backward();
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    assert_eq!(x.grad().unwrap().item(), 31.);
    assert_eq!(outputs[0].grad().unwrap().item(), 8.);
    assert_eq!(outputs[1].grad().unwrap().item(), 5.);
    assert_eq!(*observed.lock().unwrap_or_else(|e| e.into_inner()), vec![vec![Some(8.), Some(5.)]; 2]);
}

#[test]
fn duplicate_and_ancestor_roots_seed_before_shared_callback_and_keep_target_identity() {
    let x = leaf(2.);
    let calls = Arc::new(AtomicUsize::new(0)); let count = calls.clone();
    let outputs = Tensor::record_many_optional(vec![Tensor::scalar(4.), Tensor::scalar(6.)], vec![x.clone()], &[true, true], move |seeds| {
        count.fetch_add(1, Ordering::Relaxed);
        assert_eq!(seeds[0].as_ref().unwrap().item(), 5.5);
        assert_eq!(seeds[1].as_ref().unwrap().item(), 4.);
        vec![Some(Tensor::scalar(23.))]
    }).unwrap();
    let total = outputs[0].add(&outputs[1]).unwrap();
    let seeds = [Tensor::scalar(2.), Tensor::scalar(-0.5), Tensor::scalar(4.), Tensor::scalar(7.)];
    let result = Tensor::vjp_many_wrt_optional(&[&outputs[0], &outputs[0], &total, &x], &[&x, &outputs[0], &x, &total], &[&seeds[0], &seeds[1], &seeds[2], &seeds[3]], false).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(result.iter().map(|g| g.as_ref().unwrap().item()).collect::<Vec<_>>(), vec![30., 5.5, 30., 4.]);
    assert_eq!(result[0].as_ref().unwrap().id(), result[2].as_ref().unwrap().id());
    assert_eq!(result[3].as_ref().unwrap().id(), seeds[2].id());
    assert!(x.grad().is_none() && outputs.iter().all(|value| value.grad().is_none()));
}

#[test]
fn selected_sibling_is_a_cutoff_not_a_dependency_on_other_outputs() {
    let x = leaf(2.);
    let calls = Arc::new(AtomicUsize::new(0)); let count = calls.clone();
    let outputs = Tensor::record_many_optional(vec![Tensor::scalar(4.), Tensor::scalar(6.)], vec![x.clone()], &[true, true], move |_| {
        count.fetch_add(1, Ordering::Relaxed);
        panic!("selected sibling query must not enter its producer");
    }).unwrap();
    let total = outputs[0].square().unwrap();
    x.detach().copy_from(&Tensor::scalar(9.)).unwrap();
    let result = total.vjp_wrt_optional(&[&outputs[0], &outputs[1]], &Tensor::scalar(1.), false).unwrap();
    assert_eq!(result[0].as_ref().unwrap().item(), 8.);
    assert!(result[1].is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(total.grad_wrt(&[&x], false).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0, "stale parent check precedes callback");
    assert!(x.grad().is_none() && total.grad().is_none());
}

#[test]
fn absent_output_cotangent_and_present_zero_keep_input_absence_distinct() {
    let x = leaf(2.); let y = leaf(3.);
    let observed = Arc::new(Mutex::new(Vec::new())); let seen = observed.clone();
    let outputs = Tensor::record_many_optional(vec![Tensor::scalar(2.), Tensor::scalar(3.)], vec![x.clone(), y.clone()], &[true, true], move |seeds| {
        seen.lock().unwrap_or_else(|e| e.into_inner()).push(seeds.iter().map(|g| g.as_ref().map(Tensor::item)).collect::<Vec<_>>());
        vec![seeds[0].clone(), seeds[1].clone()]
    }).unwrap();
    let one = Tensor::scalar(1.); let zero = Tensor::scalar(0.);
    let missing = outputs[0].vjp_wrt_optional(&[&x, &y], &one, false).unwrap();
    assert_eq!(missing[0].as_ref().unwrap().item(), 1.); assert!(missing[1].is_none());
    let present = Tensor::vjp_many_wrt_optional(&[&outputs[0], &outputs[1]], &[&x, &y], &[&one, &zero], false).unwrap();
    assert_eq!(present[1].as_ref().unwrap().item(), 0.);
    assert!(x.grad().is_none() && y.grad().is_none());
    outputs[0].backward(); assert!(y.grad().is_none());
    outputs[0].add(&outputs[1].mul(&zero).unwrap()).unwrap().backward();
    assert_eq!(x.grad().unwrap().item(), 2.); assert_eq!(y.grad().unwrap().item(), 0.);
    assert_eq!(*observed.lock().unwrap_or_else(|e| e.into_inner()), vec![vec![Some(1.), None], vec![Some(1.), Some(0.)], vec![Some(1.), None], vec![Some(1.), Some(0.)]]);
}

#[test]
fn malformed_late_input_contribution_does_not_publish_any_gradient_slot() {
    for invalid in 0..3 {
        let x = leaf(2.); let y = leaf(3.);
        x.square().unwrap().backward(); y.backward();
        let held_x = x.grad().unwrap(); let held_y = y.grad().unwrap();
        let outputs = Tensor::record_many_optional(vec![Tensor::scalar(4.), Tensor::scalar(6.)], vec![x.clone(), y.clone()], &[true, true], move |seeds| {
            let first = seeds[0].clone();
            match invalid {
                0 => vec![first],
                1 => vec![first, Some(Tensor::ones(&[2]))],
                _ => vec![first, Some(Tensor::from_vec_i64(vec![1], &[]).unwrap())],
            }
        }).unwrap();
        let good_calls = Arc::new(AtomicUsize::new(0)); let count = good_calls.clone();
        let good = Tensor::scalar(8.).record_fn(vec![x.clone()], move |g| { count.fetch_add(1, Ordering::Relaxed); vec![g.clone()] });
        let shared = outputs[0].add(&outputs[1]).unwrap();
        let total = shared.add(&good).unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| total.backward())).is_err());
        assert_eq!(good_calls.load(Ordering::Relaxed), 1, "a valid earlier contribution was staged before the late fault");
        for value in [&total, &shared, &good, &outputs[0], &outputs[1]] { assert!(value.grad().is_none()); }
        assert_eq!((x.grad().unwrap().id(), y.grad().unwrap().id()), (held_x.id(), held_y.id()));
        assert_eq!((held_x.item(), held_y.item()), (4., 1.));
        assert!(total.grad_wrt(&[&x, &y], false).is_err());
        assert_eq!((x.grad().unwrap().id(), y.grad().unwrap().id()), (held_x.id(), held_y.id()));
    }
}

#[test]
fn stale_unused_parent_rejects_before_callback_without_partial_publication() {
    let x = leaf(2.); let y = leaf(3.);
    x.backward(); y.backward(); let held_x = x.grad().unwrap(); let held_y = y.grad().unwrap();
    let calls = Arc::new(AtomicUsize::new(0)); let count = calls.clone();
    let outputs = Tensor::record_many_optional(vec![Tensor::scalar(4.), Tensor::scalar(6.)], vec![x.clone(), y.clone()], &[true, true], move |seeds| {
        count.fetch_add(1, Ordering::Relaxed); vec![seeds[0].clone(), None]
    }).unwrap();
    let good = x.square().unwrap(); let total = outputs[0].add(&good).unwrap();
    y.detach().copy_from(&Tensor::scalar(11.)).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| total.backward())).is_err());
    assert!(total.grad_wrt(&[&x], false).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(total.grad().is_none() && good.grad().is_none() && outputs.iter().all(|v| v.grad().is_none()));
    assert_eq!((x.grad().unwrap().id(), y.grad().unwrap().id()), (held_x.id(), held_y.id()));
    assert_eq!((held_x.item(), held_y.item()), (1., 1.));
}

#[test]
fn mixed_output_types_and_nondifferentiable_metadata_keep_slot_order_and_precision() {
    let x = Tensor::from_vec_f64(vec![2.], &[]).unwrap().requires_grad_(true).unwrap();
    let calls = Arc::new(AtomicUsize::new(0)); let count = calls.clone();
    let outputs = Tensor::record_many_optional(vec![Tensor::scalar(4.), Tensor::from_vec_f64(vec![6.], &[]).unwrap(), Tensor::from_vec_i64(vec![7], &[]).unwrap()], vec![x.clone()], &[true, true, false], move |seeds| {
        count.fetch_add(1, Ordering::Relaxed);
        assert_eq!(seeds.len(), 3); assert!(seeds[2].is_none());
        let left = seeds[0].as_ref().map_or(0., |v| { assert_eq!(v.dtype(), DType::F32); v.item() as f64 });
        let right = seeds[1].as_ref().map_or(0., |v| { assert_eq!(v.dtype(), DType::F64); v.to_vec_f64()[0] });
        vec![Some(Tensor::from_vec_f64(vec![2. * left + 3. * right], &[]).unwrap())]
    }).unwrap();
    assert!(!outputs[2].requires_grad()); assert_eq!(outputs[2].to_vec_i64(), vec![7]);
    let seed32 = Tensor::scalar(1.5); let seed64 = Tensor::from_vec_f64(vec![16_777_217.], &[]).unwrap();
    let gradient = Tensor::vjp_many_wrt_optional(&[&outputs[0], &outputs[1]], &[&x], &[&seed32, &seed64], false).unwrap().remove(0).unwrap();
    assert_eq!(gradient.dtype(), DType::F64); assert_eq!(gradient.to_vec_f64(), vec![50_331_654.]);
    assert_eq!(calls.load(Ordering::Relaxed), 1); assert!(x.grad().is_none());
}

#[test]
fn empty_output_cotangent_is_present_instead_of_materialized_absence() {
    let x = leaf(2.);
    let seen = Arc::new(Mutex::new(Vec::new())); let observed = seen.clone();
    let outputs = Tensor::record_many_optional(vec![Tensor::zeros(&[0, 2]), Tensor::scalar(3.)], vec![x.clone()], &[true, true], move |seeds| {
        observed.lock().unwrap_or_else(|e| e.into_inner()).push((seeds[0].as_ref().map(|g| g.shape().to_vec()), seeds[1].as_ref().map(Tensor::item)));
        if let Some(empty) = &seeds[0] { assert_eq!(empty.numel(), 0); }
        vec![Some(Tensor::scalar(seeds[1].as_ref().map_or(0., Tensor::item)))]
    }).unwrap();
    outputs[0].sum().backward();
    assert_eq!(x.grad().unwrap().item(), 0.);
    assert_eq!(outputs[1].grad_wrt(&[&x], false).unwrap()[0].item(), 1.);
    assert_eq!(*seen.lock().unwrap_or_else(|e| e.into_inner()), vec![(Some(vec![0, 2]), None), (None, Some(1.))]);
}
