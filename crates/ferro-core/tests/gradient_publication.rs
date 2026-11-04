use std::sync::{Arc, Mutex};
use ferro_core::{DType, Tensor};
use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};

fn values(data: &[f64], shape: &[usize], dtype: DType) -> Tensor {
    Tensor::from_vec_f64(data.to_vec(), shape).unwrap().to_dtype(dtype)
}

fn full(shape: &[usize], value: f64, dtype: DType) -> Tensor {
    values(&vec![value; shape.iter().product()], shape, dtype)
}

fn input(dtype: DType, layout: usize) -> Tensor {
    let base = values(&(1..13).map(|value| value as f64).collect::<Vec<_>>(), &[2, 6], dtype);
    let value = match layout {
        0 => base.narrow(1, 0, 3).unwrap().clone_tensor(false).unwrap(),
        1 => base.narrow(1, 0, 3).unwrap().clone_tensor(false).unwrap().transpose(0, 1).unwrap(),
        _ => base.slice(1, 0, 6, 2).unwrap(),
    };
    value.detach().requires_grad_(true).unwrap()
}

#[test]
fn ordinary_publication_matches_120_torch_callback_prior_slot_and_layout_cases() {
    let mut count = 0;
    for dtype in [DType::F32, DType::F64] { for layout in 0..3 { for callback in 0..4 { for prior in 0..5 {
        let x = input(dtype, layout);
        let original = x.to_vec_f64();
        let owner = full(x.shape(), 3.0, dtype).requires_grad_(true).unwrap();
        let previous = match prior {
            0 => None,
            1 => Some(full(x.shape(), 7.0, dtype)),
            2 => Some(full(x.shape(), 7.0, dtype).requires_grad_(true).unwrap()),
            3 => Some(owner.mul(&full(&[], 2.0, dtype)).unwrap()),
            _ => Some(full(&[x.shape()[1], x.shape()[0]], 7.0, dtype).transpose(0, 1).unwrap()),
        };
        x.set_leaf_grad(previous.clone()).unwrap();
        let old_flags = previous.as_ref().map(|value| (value.requires_grad(), value.is_leaf(), value.strides().to_vec(), value._version()));
        let held = x.clone_tensor(true).unwrap().detach().requires_grad_(true).unwrap();
        let source = x.clone();
        let modes = Arc::new(Mutex::new(Vec::new()));
        let seen = modes.clone();
        let output = full(&[], 1.0, dtype).record_fn(vec![x.clone()], move |_| {
            seen.lock().unwrap().push(is_grad_enabled());
            vec![match callback {
                0 => source.clone(),
                1 => source.detach(),
                2 => with_grad_enabled(true, || source.mul(&full(&[], 1.0, dtype)).unwrap()),
                _ => held.clone(),
            }]
        });
        output.backward();
        let first = x.grad().unwrap();
        let base = if prior == 0 { 0.0 } else if prior == 3 { 6.0 } else { 7.0 };
        assert_eq!(first.to_vec_f64(), original.iter().map(|value| value + base).collect::<Vec<_>>());
        if let Some(previous) = &previous {
            assert_eq!(first.id(), previous.id());
            let (requires, leaf, stride, version) = old_flags.as_ref().unwrap();
            assert_eq!((first.requires_grad(), first.is_leaf()), (*requires, *leaf));
            assert_eq!(first.strides(), stride); assert_eq!(first._version(), version + 1);
        } else {
            assert!(!first.requires_grad() && first.is_leaf());
            assert_ne!(first.id(), x.id());
            assert_eq!(first.strides(), if layout == 1 { &[1, 3][..] } else { &[3, 1][..] });
            if callback == 0 || callback == 3 { assert_ne!(first._storage_ptr(), x._storage_ptr()); }
            if callback == 1 && layout != 2 { assert_eq!(first._storage_ptr(), x._storage_ptr()); }
        }
        let version = first._version();
        output.backward();
        assert_eq!(x.grad().unwrap().id(), first.id());
        assert_eq!(first._version(), version + 1);
        assert_eq!(first.to_vec_f64(), original.iter().map(|value| 2.0 * value + base).collect::<Vec<_>>());
        assert_eq!(*modes.lock().unwrap(), vec![false, false]);
        if prior == 3 {
            let derivative = first.sum().grad_wrt(&[&owner], false).unwrap().remove(0);
            assert_eq!(derivative.to_vec_f64(), vec![2.0; owner.numel()]);
        }
        assert!(owner.grad().is_none()); assert!(is_grad_enabled()); count += 1;
    } } } }
    assert_eq!(count, 120);
}

#[test]
fn upstream_callbacks_keep_live_cotangents_but_public_slots_are_detached() {
    for dtype in [DType::F32, DType::F64] { for enabled in [false, true] {
        let x = full(&[], 2.0, dtype).requires_grad_(true).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new())); let captured = seen.clone();
        let upstream = full(&[], 2.0, dtype).record_fn(vec![x.clone()], move |g| {
            captured.lock().unwrap().push((is_grad_enabled(), g.requires_grad(), g.is_leaf()));
            vec![g.clone()]
        });
        let parent = upstream.clone();
        let output = full(&[], 2.0, dtype).record_fn(vec![upstream.clone()], move |_| vec![parent.clone()]);
        with_grad_enabled(enabled, || { output.backward(); assert_eq!(is_grad_enabled(), enabled); });
        assert_eq!(*seen.lock().unwrap(), vec![(false, true, false)]);
        for value in [&x, &upstream] {
            let gradient = value.grad().unwrap();
            assert!(!gradient.requires_grad() && gradient.is_leaf()); assert_ne!(gradient.id(), value.id());
        }
    } }
}

#[test]
fn shared_manual_slots_accumulate_sequentially_without_replacing_graph_history() {
    for dtype in [DType::F32, DType::F64] { for live in [false, true] {
        let a = full(&[2, 3], 1.0, dtype).requires_grad_(true).unwrap();
        let b = full(&[3, 2], 1.0, dtype).requires_grad_(true).unwrap();
        let owner = full(&[2, 3], 3.0, dtype).requires_grad_(live).unwrap();
        let shared = if live { owner.mul(&full(&[], 2.0, dtype)).unwrap() } else { owner.clone_tensor(true).unwrap() };
        let transposed = shared.transpose(0, 1).unwrap();
        a.set_leaf_grad(Some(shared.clone())).unwrap(); b.set_leaf_grad(Some(transposed.clone())).unwrap();
        let version = shared._version();
        a.mul(&full(&[], 2.0, dtype)).unwrap().sum().add(&b.mul(&full(&[], 3.0, dtype)).unwrap().sum()).unwrap().backward();
        assert_eq!(a.grad().unwrap().id(), shared.id()); assert_eq!(b.grad().unwrap().id(), transposed.id());
        assert_eq!(shared.to_vec_f64(), vec![if live { 11.0 } else { 8.0 }; 6]);
        assert_eq!(shared._version(), version + 2); assert_eq!(shared.requires_grad(), live);
        if live { assert_eq!(shared.sum().grad_wrt(&[&owner], false).unwrap()[0].to_vec_f64(), vec![2.0; 6]); }
    } }
}

#[test]
fn late_invalid_overlapping_slot_does_not_mutate_earlier_slots_or_poison_locks() {
    let a = full(&[2, 2], 1.0, DType::F32).requires_grad_(true).unwrap();
    let b = full(&[2, 2], 1.0, DType::F32).requires_grad_(true).unwrap();
    let first = full(&[2, 2], 7.0, DType::F32);
    let invalid = Tensor::scalar(5.0).expand(&[2, 2]).unwrap();
    a.set_leaf_grad(Some(first.clone())).unwrap(); b.set_leaf_grad(Some(invalid.clone())).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.add(&b).unwrap().sum().backward()));
    assert!(result.is_err()); assert!(is_grad_enabled());
    assert_eq!(a.grad().unwrap().id(), first.id()); assert_eq!(first.to_vec(), vec![7.0; 4]);
    assert_eq!(b.grad().unwrap().id(), invalid.id()); assert_eq!(invalid.to_vec(), vec![5.0; 4]);
    b.set_leaf_grad(Some(Tensor::zeros(&[2, 2]))).unwrap();
    a.add(&b).unwrap().sum().backward();
    assert_eq!(first.to_vec(), vec![8.0; 4]);
}

#[test]
fn concurrent_independent_leaves_serialize_shared_gradient_cells() {
    for dtype in [DType::F32, DType::F64] {
        let shared = full(&[2, 3], 0.0, dtype);
        let a = full(&[2, 3], 1.0, dtype).requires_grad_(true).unwrap();
        let b = full(&[3, 2], 1.0, dtype).requires_grad_(true).unwrap();
        let transposed = shared.transpose(0, 1).unwrap();
        a.set_leaf_grad(Some(shared.clone())).unwrap();
        b.set_leaf_grad(Some(transposed.clone())).unwrap();
        let start = Arc::new(std::sync::Barrier::new(2));
        std::thread::scope(|scope| {
            for (leaf, increment) in [(&a, 2.0), (&b, 3.0)] {
                let start = start.clone();
                scope.spawn(move || {
                    let loss = leaf.mul(&full(&[], increment, dtype)).unwrap().sum();
                    for _ in 0..100 { start.wait(); loss.backward(); }
                });
            }
        });
        assert_eq!(shared.to_vec_f64(), vec![500.0; 6]);
        assert_eq!(shared._version(), 200);
        assert_eq!(a.grad().unwrap().id(), shared.id());
        assert_eq!(b.grad().unwrap().id(), transposed.id());
    }
}

#[test]
fn live_seed_reaches_raw_callbacks_unchanged_but_never_becomes_a_live_public_slot() {
    for dtype in [DType::F32, DType::F64] { for live in [false, true] {
        let x = full(&[2], 1.0, dtype).requires_grad_(true).unwrap();
        let seed = full(&[2], 3.0, dtype).requires_grad_(live).unwrap();
        let seed_id = seed.id();
        let y = full(&[2], 1.0, dtype).record_fn(vec![x.clone()], move |g| {
            assert_eq!(g.id(), seed_id); assert_eq!(g.requires_grad(), live);
            assert!(!is_grad_enabled()); vec![g.clone()]
        });
        y.backward_with(&seed);
        for value in [&x, &y] {
            let gradient = value.grad().unwrap();
            assert_eq!(gradient.to_vec_f64(), vec![3.0; 2]);
            assert!(!gradient.requires_grad() && gradient.is_leaf());
            assert_ne!(gradient._storage_ptr(), seed._storage_ptr());
        }
        assert!(seed.grad().is_none()); assert!(is_grad_enabled());
    } }
}
