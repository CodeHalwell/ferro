use ferro_core::autograd::with_grad_enabled;
use ferro_core::{DType, Tensor};

fn values(dtype: DType, requires_grad: bool) -> Tensor {
    let tensor = Tensor::from_vec_f64(vec![2., 3.], &[2]).unwrap().to_dtype(dtype);
    tensor.set_requires_grad(requires_grad).unwrap();
    tensor
}

#[test]
fn mutable_leaf_flags_preserve_identity_shared_handles_and_existing_slots() {
    for dtype in [DType::F32, DType::F64] {
        let x = values(dtype, true);
        let alias = x.clone();
        let id = x.id();
        let version = x._version();
        let loss = x.square().unwrap().sum();
        loss.backward();
        let held = x.grad().unwrap();
        x.set_requires_grad(false).unwrap();
        assert!(!alias.requires_grad());
        assert_eq!(x.id(), id);
        assert_eq!(x.grad().unwrap().id(), held.id());
        loss.backward();
        assert_eq!(x.grad().unwrap().id(), held.id());
        x.set_requires_grad(true).unwrap();
        assert!(alias.requires_grad());
        assert_eq!(x.id(), id);
        loss.backward();
        assert_eq!(x.grad().unwrap().to_vec_f64(), vec![8., 12.]);
        assert_eq!(held.id(), x.grad().unwrap().id());
        assert_eq!(held.to_vec_f64(), vec![8., 12.]);
        assert_eq!(x._version(), version);
    }
}

#[test]
fn late_enabled_operands_never_acquire_old_edges_but_saved_values_stay_checked() {
    for dtype in [DType::F32, DType::F64] {
        let inactive = values(dtype, false);
        let active = values(dtype, true);
        let loss = inactive.mul(&active).unwrap().sum();
        inactive.set_requires_grad(true).unwrap();
        assert!(!loss.depends_on(&inactive));
        assert!(loss.depends_on(&active));
        assert!(!loss.has_grad_leaf_except(&[&active]));
        assert!(loss.has_current_grad_leaf_except(&[&active]));
        let gradients = loss.grad_wrt(&[&inactive, &active], false).unwrap();
        assert_eq!(gradients[0].to_vec_f64(), vec![0., 0.]);
        assert_eq!(gradients[1].to_vec_f64(), vec![2., 3.]);
        loss.backward();
        assert!(inactive.grad().is_none());
        assert_eq!(active.grad().unwrap().to_vec_f64(), vec![2., 3.]);
        let held = active.grad().unwrap();
        with_grad_enabled(false, || inactive.copy_leaf_from(&Tensor::zeros(&[2]).to_dtype(dtype))).unwrap();
        assert!(loss.grad_wrt(&[&active], false).is_err());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loss.backward())).is_err());
        assert_eq!(active.grad().unwrap().id(), held.id());
    }
}

#[test]
fn late_base_flags_do_not_create_view_edges_in_either_recording_mode() {
    for dtype in [DType::F32, DType::F64] {
        for grad_enabled in [false, true] {
            let base = values(dtype, false);
            let view = with_grad_enabled(grad_enabled, || base.reshape(&[1, 2])).unwrap();
            base.set_requires_grad(true).unwrap();
            assert!(view.requires_grad() && view.is_leaf());
            let loss = view.square().unwrap().sum();
            assert!(loss.requires_grad());
            let optional = loss.vjp_wrt_optional(&[&base, &view], &Tensor::ones(&[]).to_dtype(dtype), false).unwrap();
            assert!(optional.iter().all(Option::is_none));
            loss.backward();
            assert!(view.grad().is_none() && base.grad().is_none());
            assert!(view.grad_wrt(&[&view], false).is_err());
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| view.backward_with(&Tensor::ones(view.shape()).to_dtype(dtype)))).is_err());
            base.set_requires_grad(false).unwrap();
            assert!(!view.requires_grad());
            base.set_requires_grad(true).unwrap();
            assert!(view.requires_grad());
            assert!(!view.square().unwrap().sum().depends_on(&view));
        }
    }
}

#[test]
fn old_view_edges_survive_own_flag_disable_when_inherited_flag_remains_true() {
    for dtype in [DType::F32, DType::F64] {
        let base = values(dtype, false);
        let view = base.reshape(&[1, 2]).unwrap();
        view.set_requires_grad(true).unwrap();
        let old = view.square().unwrap().sum();
        base.set_requires_grad(true).unwrap();
        view.set_requires_grad(false).unwrap();
        assert!(view.requires_grad() && view.is_leaf());
        assert!(old.depends_on(&view) && !old.depends_on(&base));
        let gradients = old.grad_wrt(&[&view, &base], false).unwrap();
        assert_eq!(gradients[0].to_vec_f64(), vec![4., 6.]);
        assert_eq!(gradients[1].to_vec_f64(), vec![0., 0.]);
        old.backward();
        assert_eq!(view.grad().unwrap().to_vec_f64(), vec![4., 6.]);
        assert!(base.grad().is_none());
        let new = view.square().unwrap().sum();
        assert!(!new.depends_on(&view));
        new.backward();
        assert_eq!(view.grad().unwrap().to_vec_f64(), vec![4., 6.]);
        assert!(view.vjp_wrt(&[&view], &Tensor::ones(view.shape()).to_dtype(dtype), false).is_err());
    }
}

#[test]
fn ultimate_base_flags_and_saved_public_provenance_are_distinct_from_edges() {
    let base = values(DType::F32, false);
    let view = base.reshape(&[1, 2]).unwrap();
    view.set_requires_grad(true).unwrap();
    let recorded = view.transpose(0, 1).unwrap();
    let frozen = with_grad_enabled(false, || view.transpose(0, 1)).unwrap();
    assert!(recorded.requires_grad());
    assert!(!frozen.requires_grad());
    let detached = recorded.detach();
    let copied = with_grad_enabled(false, || recorded.clone_tensor(true)).unwrap();
    view.set_requires_grad(false).unwrap();
    assert!(recorded.has_grad_leaf_except(&[]));
    assert!(!recorded.has_current_grad_leaf_except(&[]));
    assert!(!recorded.has_grad_leaf_except(&[&view]));
    assert!(recorded.depends_on(&view));
    base.set_requires_grad(true).unwrap();
    assert!(frozen.requires_grad());
    assert!(!detached.requires_grad() && !copied.requires_grad());
    let from_blocked = frozen.square().unwrap();
    assert!(from_blocked.has_grad_leaf_except(&[]));
    assert!(!from_blocked.depends_on(&frozen));
    base.set_requires_grad(false).unwrap();
    assert!(!frozen.requires_grad());
    assert!(from_blocked.has_grad_leaf_except(&[]));
    assert!(!from_blocked.has_grad_leaf_except(&[&frozen]));
}

#[test]
fn invalid_flags_and_missing_direct_edges_reject_before_slot_changes() {
    for dtype in [DType::I64, DType::Bool, DType::F16, DType::BF16] {
        let x = values(dtype, false);
        let id = x.id();
        assert!(x.set_requires_grad(true).is_err());
        assert!(!x.requires_grad());
        assert_eq!(x.id(), id);
    }
    let x = values(DType::F64, true);
    let out = x.square().unwrap().sum();
    out.set_requires_grad(true).unwrap();
    assert!(out.set_requires_grad(false).is_err());
    assert!(out.requires_grad());
    let constant = Tensor::ones(&[]);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| constant.backward())).is_err());
    assert!(constant.grad().is_none());
    assert!(x.grad().is_none());
}
