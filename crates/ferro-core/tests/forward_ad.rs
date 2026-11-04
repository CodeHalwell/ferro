use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use ferro_core::forward_ad::{is_active, DualLevel};
use ferro_core::Tensor;

fn leaf(values: &[f32]) -> Tensor { Tensor::from_vec(values.to_vec(), &[values.len()]).unwrap().requires_grad_(true).unwrap() }
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() < 3e-6, "{a} != {b}"); }
}

#[test]
fn dual_flows_through_no_grad_chain_without_creating_reverse_edges() {
    let x = leaf(&[0.2, -0.4]);
    let v = leaf(&[0.3, 0.7]);
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let y = with_grad_enabled(false, || x.mul(&x).unwrap().tanh().exp());
    let dy = level.tangent(&y).unwrap().unwrap();
    assert!(!y.requires_grad() && y.is_leaf() && !dy.requires_grad());
    let expected = [0.2f32, -0.4].iter().zip([0.3, 0.7]).map(|(&x, v)| { let t = (x*x).tanh(); 2.*x*v*(1.-t*t)*t.exp() }).collect::<Vec<_>>();
    close(&dy.to_vec(), &expected);
    assert!(level.tangent(&y.detach()).unwrap().is_none());
    drop(level);
    assert!(is_grad_enabled() && !is_active());
    assert!(x.grad().is_none() && v.grad().is_none());
}

#[test]
fn dual_values_reverse_derivatives_and_output_versions_share_the_existing_op() {
    let x = leaf(&[0.2, -0.4]);
    let v = leaf(&[0.3, 0.7]);
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let y = x.mul(&x).unwrap();
    let dy = level.tangent(&y).unwrap().unwrap();
    drop(level);
    close(&dy.to_vec(), &[0.12, -0.56]);
    let grads = dy.sum().grad_wrt(&[&x, &v], true).unwrap();
    close(&grads[0].to_vec(), &[0.6, 1.4]);
    close(&grads[1].to_vec(), &[0.4, -0.8]);
    close(&grads[0].sum().grad_wrt(&[&v], false).unwrap()[0].to_vec(), &[2., 2.]);
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let y = x.tanh();
    let dy = level.tangent(&y).unwrap().unwrap();
    drop(level);
    y.detach().copy_from(&Tensor::zeros(&[2])).unwrap();
    assert!(y.sum().grad_wrt(&[&x], false).is_err());
    assert!(dy.sum().grad_wrt(&[&x, &v], false).is_err());
    assert!(x.grad().is_none() && v.grad().is_none());
    let constant = Tensor::from_vec(vec![0.2, 0.4], &[2]).unwrap();
    let level = DualLevel::new(&[&constant], &[&v]).unwrap();
    let primal = constant.tanh();
    let direction = level.tangent(&primal).unwrap().unwrap();
    drop(level);
    assert!(!primal.requires_grad() && direction.requires_grad());
    primal.copy_from(&Tensor::zeros(&[2])).unwrap();
    assert!(direction.sum().grad_wrt(&[&v], false).is_err());
    for kind in ["sqrt", "sin", "cos"] {
        let input = Tensor::from_vec(vec![0.3, 0.5], &[2]).unwrap();
        let level = DualLevel::new(&[&input], &[&v]).unwrap();
        let primal = match kind { "sqrt" => input.sqrt(), "sin" => input.sin().unwrap(), _ => input.cos().unwrap() };
        let direction = level.tangent(&primal).unwrap().unwrap();
        drop(level);
        input.copy_from(&Tensor::zeros(&[2])).unwrap();
        primal.copy_from(&Tensor::zeros(&[2])).unwrap();
        assert!(direction.sum().grad_wrt(&[&v], false).is_ok(), "{kind}");
    }
}

#[test]
fn local_opaque_callback_fallback_preserves_numeric_no_grad_and_validates_errors() {
    let x = Tensor::from_vec(vec![0.2, -0.4], &[2]).unwrap();
    let v = leaf(&[0.3, 0.7]);
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let y = with_grad_enabled(false, || Tensor::from_vec(vec![0.4, -0.8], &[2]).unwrap().record_fn(vec![x.clone()], |g| vec![g.mul(&Tensor::scalar(2.)).unwrap()]));
    close(&level.tangent(&y).unwrap().unwrap().to_vec(), &[0.6, 1.4]);
    assert!(!y.requires_grad());
    drop(level);
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let y = Tensor::from_vec(vec![0.4, -0.8], &[2]).unwrap().record_fn(vec![x.clone()], |g| vec![g.mul(&Tensor::scalar(2.)).unwrap()]);
    let tangent = level.tangent(&y).unwrap().unwrap();
    drop(level);
    x.copy_from(&Tensor::zeros(&[2])).unwrap();
    assert!(tangent.sum().grad_wrt(&[&v], false).is_err());
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let bad = Tensor::zeros(&[2]).record_fn(vec![x.clone()], |_| vec![Tensor::scalar(0.)]);
    assert!(level.tangent(&bad).is_err());
    drop(level);
    let active = x.requires_grad_(true).unwrap();
    let level = DualLevel::new(&[&active], &[&v]).unwrap();
    let opaque = Tensor::zeros(&[2]).record_fn(vec![active.clone()], |g| vec![g.clone()]);
    assert!(level.tangent(&opaque).is_err());
    drop(level);
    assert!(!is_active() && is_grad_enabled());
}

#[test]
fn dual_level_rejects_nested_mutation_and_restores_after_unwind() {
    let x = Tensor::ones(&[2]);
    let v = Tensor::ones(&[2]);
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    assert!(DualLevel::new(&[&x], &[&v]).is_err());
    assert!(x.copy_from(&Tensor::zeros(&[2])).is_err());
    assert_eq!(x.to_vec(), vec![1., 1.]);
    drop(level);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _level = DualLevel::new(&[&x], &[&v]).unwrap();
        panic!("test unwind");
    })).is_err());
    assert!(!is_active());
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    assert_eq!(level.tangent(&x).unwrap().unwrap().id(), v.id());
    assert!(DualLevel::new(&[&x, &x], &[&v, &v]).is_err());
    let other = leaf(&[0.2, 0.4]);
    let loss = other.mul(&other).unwrap().sum();
    assert!(loss.grad_wrt(&[&other], false).is_err());
    assert!(loss.grad_wrt(&[&other], true).is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loss.backward())).is_err());
    assert!(other.grad().is_none() && loss.grad().is_none());
    let captured = ferro_core::capture(|| x.mul(&x).unwrap());
    assert!(level.tangent(&captured).is_err());
}

#[test]
fn direct_view_cat_cast_empty_and_broadcast_matmul_rules_keep_real_tangents() {
    let x = leaf(&[0.2, 0.4, -0.3, 0.5]).reshape(&[2, 2]).unwrap();
    let v = leaf(&[0.3, 0.7, 0.1, -0.2]).reshape(&[2, 2]).unwrap();
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let view = x.permute(&[1, 0]).unwrap().unsqueeze(0).unwrap().expand(&[3, 2, 2]).unwrap();
    let dy = level.tangent(&view).unwrap().unwrap();
    close(&dy.to_vec(), &[0.3, 0.1, 0.7, -0.2, 0.3, 0.1, 0.7, -0.2, 0.3, 0.1, 0.7, -0.2]);
    let joined = Tensor::cat(&[x.clone(), x.narrow(0, 0, 0).unwrap()], 0).unwrap();
    close(&level.tangent(&joined).unwrap().unwrap().to_vec(), &v.to_vec());
    let cast = x.to_dtype_autograd(ferro_core::DType::F64, false).unwrap();
    assert_eq!(level.tangent(&cast).unwrap().unwrap().dtype(), ferro_core::DType::F64);
    let y = x.matmul(&Tensor::ones(&[2, 1])).unwrap();
    close(&level.tangent(&y).unwrap().unwrap().to_vec(), &[1., -0.1]);
}

#[test]
fn constant_multiply_and_divide_keep_structural_zero_derivatives() {
    for divide in [false, true] {
        let x = leaf(&[f32::INFINITY, f32::NAN]);
        let v = Tensor::from_vec(vec![0.3, -0.7], &[2]).unwrap();
        let coefficient = Tensor::scalar(2.).requires_grad_(true).unwrap();
        let level = DualLevel::new(&[&x], &[&v]).unwrap();
        let output = if divide { x.div(&coefficient).unwrap() } else { x.mul(&coefficient).unwrap() };
        let tangent = level.tangent(&output).unwrap().unwrap();
        drop(level);
        close(&tangent.to_vec(), if divide { &[0.15, -0.35] } else { &[0.6, -1.4] });
        let derivatives = tangent.sum().grad_wrt(&[&x, &coefficient], true).unwrap();
        assert!(derivatives[0].is_structural_zero());
        assert_eq!(derivatives[0].requires_grad(), divide);
        if divide { assert!(derivatives[1].item().is_nan()); } else { close(&derivatives[1].to_vec(), &[-0.4]); }
    }
}

#[test]
fn functional_duals_ignore_factory_leaves_without_mutating_them_and_unwrap_primal_aliases() {
    for live in [false, true] {
        for enabled in [false, true] {
            let x = Tensor::from_vec(vec![0.2, -0.4], &[2]).unwrap().requires_grad_(live).unwrap();
            let v = Tensor::from_vec(vec![0.3, 0.7], &[2]).unwrap().requires_grad_(live).unwrap();
            let captured = leaf(&[0.5, 0.5]);
            let (primal, tangent, factory, auxiliary, identity, captured_aux) = with_grad_enabled(enabled, || {
                let level = DualLevel::functional(&[&x], &[&v]).unwrap();
                let factory = leaf(&[0.5, 0.5]);
                let view = factory.reshape(&[2]).unwrap();
                let result = x.mul(&view).unwrap();
                let tangent = level.tangent(&result).unwrap().unwrap();
                let primal = level.primal(&result, false).unwrap();
                let auxiliary = level.primal(&factory, true).unwrap();
                let identity = level.primal(&x, false).unwrap();
                let captured_aux = level.primal(&captured, true).unwrap();
                (primal, tangent, factory, auxiliary, identity, captured_aux)
            });
            assert!(factory.requires_grad());
            assert!(!auxiliary.requires_grad());
            assert_eq!(captured_aux.id(), captured.id());
            assert_eq!(identity.requires_grad(), live);
            assert_eq!(identity.is_leaf(), !enabled || !live);
            assert_ne!(identity.id(), x.id());
            assert_eq!(primal.requires_grad(), enabled && live);
            assert_eq!(tangent.requires_grad(), enabled && live);
            close(&tangent.to_vec(), &[0.15, 0.35]);
            if tangent.requires_grad() {
                let gradients = tangent.sum().grad_wrt(&[&v, &factory], true).unwrap();
                close(&gradients[0].to_vec(), &[0.5, 0.5]);
                close(&gradients[1].to_vec(), &[0., 0.]);
                assert!(!gradients[0].requires_grad() && !gradients[1].requires_grad());
                factory.detach().copy_from(&Tensor::zeros(&[2])).unwrap();
                assert!(tangent.sum().grad_wrt(&[&v], true).is_err());
            }
            if live && !enabled { assert!(identity.vjp_wrt(&[&x], &Tensor::ones(&[2]), false).is_err()); }
            assert!(factory.grad().is_none() && x.grad().is_none() && v.grad().is_none());
        }
    }
    // The lower-level dual API does not apply function-transform leaf filtering.
    let x = Tensor::ones(&[2]);
    let v = Tensor::ones(&[2]);
    let level = DualLevel::new(&[&x], &[&v]).unwrap();
    let factory = leaf(&[0.5, 0.5]);
    let result = x.mul(&factory).unwrap();
    let tangent = level.tangent(&result).unwrap().unwrap();
    drop(level);
    close(&tangent.sum().grad_wrt(&[&factory], false).unwrap()[0].to_vec(), &[1., 1.]);
    // Temporary factories must not mask unrelated results if their allocations
    // would otherwise be reused during a long-lived functional level.
    let x = leaf(&[0.2, 0.4]);
    let level = DualLevel::functional(&[&x], &[&v]).unwrap();
    for _ in 0..128 {
        let temporary = leaf(&[0.5, 0.5]);
        assert!(!level.primal(&temporary, true).unwrap().requires_grad());
    }
    let result = x.mul(&x).unwrap();
    let tangent = level.tangent(&result).unwrap().unwrap();
    drop(level);
    close(&tangent.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec(), &[2., 2.]);
    assert!(!is_active() && is_grad_enabled());
}

#[test]
fn private_graph_vjp_composes_with_duals_and_private_aliases_without_changing_public_detach() {
    use ferro_core::forward_ad::{prepare_alias, post_alias, current_tangent};
    for input_grad in [false, true] {
        for direction_grad in [false, true] {
            let x = Tensor::from_vec(vec![0.2, -0.4], &[2]).unwrap().requires_grad_(input_grad).unwrap();
            let v = Tensor::from_vec(vec![0.3, 0.7], &[2]).unwrap().requires_grad_(direction_grad).unwrap();
            let level = DualLevel::functional(&[&x], &[&v]).unwrap();
            let prepared = if input_grad { x.reshape(&[2]).unwrap() } else { x.detach().requires_grad_(true).unwrap() };
            prepare_alias(&x, &prepared).unwrap();
            assert!(prepare_alias(&x, &Tensor::zeros(&[2])).is_err());
            let loss = prepared.mul(&prepared).unwrap().sum();
            assert!(loss.grad_wrt(&[&prepared], true).is_err());
            let derivative = loss.vjp_wrt_forward(&[&prepared], &Tensor::scalar(1.)).unwrap().remove(0).unwrap();
            let tangent = current_tangent(&derivative).unwrap().unwrap();
            close(&derivative.to_vec(), &[0.4, -0.8]);
            close(&tangent.to_vec(), &[0.6, 1.4]);
            assert!(level.tangent(&derivative.detach()).unwrap().is_none());
            let result = if input_grad { derivative.clone() } else { derivative.detach() };
            let tangent = if input_grad || direction_grad { tangent } else { tangent.detach() };
            post_alias(&derivative, &result, Some(&tangent)).unwrap();
            assert_eq!(level.tangent(&result).unwrap().unwrap().id(), tangent.id());
            let output = level.primal(&result, false).unwrap();
            drop(level);
            assert_eq!(output.requires_grad(), input_grad);
            assert_eq!(tangent.requires_grad(), input_grad || direction_grad);
            if direction_grad { close(&tangent.sum().grad_wrt(&[&v], false).unwrap()[0].to_vec(), &[2., 2.]); }
            assert!(x.grad().is_none() && v.grad().is_none());
            assert!(loss.vjp_wrt_forward(&[&prepared], &Tensor::scalar(1.)).is_err());
        }
    }
    let x = leaf(&[0.2, -0.4]);
    let v = leaf(&[0.3, 0.7]);
    let w = Tensor::scalar(0.5).requires_grad_(true).unwrap();
    let level = DualLevel::functional(&[&x], &[&v]).unwrap();
    let prepared = x.reshape(&[2]).unwrap();
    prepare_alias(&x, &prepared).unwrap();
    let coefficient = with_grad_enabled(false, || prepared.mul(&prepared).unwrap().tanh());
    let loss = prepared.mul(&coefficient).unwrap().mul(&w).unwrap().sum();
    let derivative = loss.vjp_wrt_forward(&[&prepared], &Tensor::scalar(1.)).unwrap().remove(0).unwrap();
    let tangent = current_tangent(&derivative).unwrap().unwrap();
    drop(level);
    close(&derivative.to_vec(), &[0.0199893415, 0.07932426]);
    close(&tangent.to_vec(), &[0.059904106, -0.2729526]);
    let gradients = tangent.sum().grad_wrt(&[&x, &v, &w], false).unwrap();
    close(&gradients[0].to_vec(), &[0., 0.]);
    close(&gradients[1].to_vec(), &[0., 0.]);
    close(&gradients[2].to_vec(), &[-0.42609698]);
}

#[test]
fn composed_no_grad_value_postprocessing_preserves_only_real_alias_flags() {
    use ferro_core::forward_ad::{prepare_alias, freeze_post, post_alias};
    for live in [false, true] {
        let x = Tensor::ones(&[2]).requires_grad_(live).unwrap();
        let v = leaf(&[0.3, 0.7]);
        let level = DualLevel::functional(&[&x], &[&v]).unwrap();
        let cutoff = ferro_core::tensor::tensor_id_watermark();
        let prepared = if live { x.reshape(&[2]).unwrap() } else { x.detach().requires_grad_(true).unwrap() };
        prepare_alias(&x, &prepared).unwrap();
        let excluded = if live { vec![] } else { vec![&prepared] };
        let view = prepared.reshape(&[2]).unwrap();
        let direction = level.tangent(&view).unwrap().unwrap();
        let output = freeze_post(&view, cutoff, &excluded).unwrap();
        let direction = freeze_post(&direction, cutoff, &excluded).unwrap();
        assert_eq!(output.requires_grad(), live);
        assert!(output.is_leaf() && direction.is_leaf() && direction.requires_grad());
        post_alias(&view, &output, Some(&direction)).unwrap();
        assert_eq!(freeze_post(&v, cutoff, &excluded).unwrap().id(), v.id());
        let computed = prepared.mul(&Tensor::scalar(2.)).unwrap();
        assert!(!freeze_post(&computed, cutoff, &excluded).unwrap().requires_grad());
        let computed_direction = level.tangent(&computed).unwrap().unwrap();
        assert!(!freeze_post(&computed_direction, cutoff, &excluded).unwrap().requires_grad());
        drop(level);
        assert!(direction.vjp_wrt(&[&v], &Tensor::ones(&[2]), false).is_err());
        if live { assert!(output.vjp_wrt(&[&x], &Tensor::ones(&[2]), false).is_err()); }
        assert!(x.grad().is_none() && v.grad().is_none());
    }
}

#[test]
fn reverse_frame_unwraps_synthetic_ancestry_and_temporary_view_bases_without_mutating_held_values() {
    use ferro_core::forward_ad::{ReverseFrameGuard, prepare_alias, current_tangent, unwrap_values};
    let x = Tensor::from_vec(vec![0.2, -0.4], &[2]).unwrap();
    let v = Tensor::from_vec(vec![0.3, 0.7], &[2]).unwrap();
    let w = Tensor::scalar(0.5).requires_grad_(true).unwrap();
    let level = DualLevel::functional(&[&x], &[&v]).unwrap();
    let frame = ReverseFrameGuard::new().unwrap();
    assert!(ReverseFrameGuard::new().is_err());
    let prepared = x.detach().requires_grad_(true).unwrap();
    prepare_alias(&x, &prepared).unwrap();
    let squared = prepared.mul(&prepared).unwrap();
    let (root_view, temporary_view) = with_grad_enabled(false, || (prepared.reshape(&[2]).unwrap(), squared.reshape(&[2]).unwrap()));
    let loss = squared.tanh().mul(&w).unwrap().sum();
    let gradient = loss.vjp_wrt_forward(&[&prepared], &Tensor::scalar(1.)).unwrap().remove(0).unwrap();
    let tangent = current_tangent(&gradient).unwrap().unwrap();
    let values = unwrap_values(&[&gradient, &tangent, &root_view, &temporary_view, &tangent, &w]).unwrap();
    assert_eq!(values[1].id(), values[4].id());
    assert_eq!(values[5].id(), w.id());
    assert!(values[0].requires_grad() && values[1].requires_grad());
    for (actual, held) in values[2..4].iter().zip([&root_view, &temporary_view]) {
        assert!(!actual.requires_grad() && actual.is_leaf());
        assert!(held.requires_grad() && held.is_leaf());
        assert_eq!(actual._storage_ptr(), held._storage_ptr());
        assert_eq!(actual.storage_offset(), held.storage_offset());
        assert_eq!(actual.strides(), held.strides());
        actual.set_requires_grad(true).unwrap();
        assert!(actual.is_leaf());
    }
    assert!(prepared.requires_grad() && squared.requires_grad());
    drop(frame);
    drop(level);
    for held in [&root_view, &temporary_view] {
        assert!(held.vjp_wrt(&[held], &Tensor::ones(&[2]), false).is_err(), "held blocked view acquired an own edge");
    }
    let dw = values[1].sum().grad_wrt(&[&w], true).unwrap().remove(0);
    close(&dw.to_vec(), &[1.8214009]);
    assert!(!dw.requires_grad());
    assert!(prepared.grad().is_none() && squared.grad().is_none() && x.grad().is_none() && w.grad().is_none());
    values[1].sum().backward();
    close(&w.grad().unwrap().to_vec(), &[1.8214009]);
    assert!(prepared.grad().is_none() && squared.grad().is_none() && x.grad().is_none());
}

#[test]
fn reverse_frame_cleanup_and_saved_versions_preserve_the_shared_engine_contract() {
    use ferro_core::forward_ad::{ReverseFrameGuard, prepare_alias, current_tangent, unwrap_values};
    let x = Tensor::from_vec(vec![0.2, -0.4], &[2]).unwrap();
    let v = Tensor::from_vec(vec![0.3, 0.7], &[2]).unwrap();
    let w = Tensor::scalar(0.5).requires_grad_(true).unwrap();
    let level = DualLevel::functional(&[&x], &[&v]).unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _frame = ReverseFrameGuard::new().unwrap();
        panic!("frame cleanup");
    })).is_err());
    let frame = ReverseFrameGuard::new().unwrap();
    let prepared = x.detach().requires_grad_(true).unwrap();
    prepare_alias(&x, &prepared).unwrap();
    let loss = prepared.mul(&w).unwrap().tanh().sum();
    let gradient = loss.vjp_wrt_forward(&[&prepared], &Tensor::scalar(1.)).unwrap().remove(0).unwrap();
    let tangent = current_tangent(&gradient).unwrap().unwrap();
    let tangent = unwrap_values(&[&tangent]).unwrap().remove(0);
    drop(frame);
    let _frame_again = ReverseFrameGuard::new().unwrap();
    drop(_frame_again);
    drop(level);
    assert!(tangent.sum().grad_wrt(&[&w], true).is_ok());
    w.detach().copy_from(&Tensor::scalar(0.7)).unwrap();
    assert!(tangent.sum().grad_wrt(&[&w], true).is_err());
    assert!(x.grad().is_none() && w.grad().is_none());
}

#[test]
fn reverse_frame_keeps_captured_edge_decisions_across_late_view_flag_changes() {
    use ferro_core::forward_ad::{ReverseFrameGuard, prepare_alias, unwrap_values};
    for edge_before in [false, true] {
        let x = Tensor::from_vec(vec![0.2, -0.4], &[2]).unwrap();
        let v = Tensor::ones(&[2]);
        let base = leaf(&[0.5, 0.6]);
        let captured = with_grad_enabled(false, || base.reshape(&[2]).unwrap());
        if edge_before { captured.set_requires_grad(true).unwrap(); }
        let level = DualLevel::functional(&[&x], &[&v]).unwrap();
        let frame = ReverseFrameGuard::new().unwrap();
        let prepared = x.detach().requires_grad_(true).unwrap();
        prepare_alias(&x, &prepared).unwrap();
        let held = prepared.mul(&captured).unwrap().sum();
        captured.set_requires_grad(!edge_before).unwrap();
        let output = unwrap_values(&[&held]).unwrap().remove(0);
        assert!(output.requires_grad());
        drop(frame);
        drop(level);
        let gradients = output.grad_wrt(&[&captured, &base], false).unwrap();
        close(&gradients[0].to_vec(), if edge_before { &[0.2, -0.4] } else { &[0., 0.] });
        close(&gradients[1].to_vec(), &[0., 0.]);
        assert!(captured.grad().is_none() && base.grad().is_none());
    }
}

#[test]
fn reverse_frame_accepts_public_grad_flag_views_without_inventing_base_edges() {
    use ferro_core::forward_ad::{ReverseFrameGuard, prepare_alias, current_tangent, unwrap_values};
    for vector_grad in [false, true] {
        let base = leaf(&[0.2, -0.4]);
        let x = with_grad_enabled(false, || base.reshape(&[2]).unwrap());
        let v = Tensor::from_vec(vec![0.3, 0.7], &[2]).unwrap().requires_grad_(vector_grad).unwrap();
        let level = DualLevel::functional(&[&x], &[&v]).unwrap();
        let frame = ReverseFrameGuard::new().unwrap();
        let prepared = x.reshape(&[2]).unwrap();
        prepare_alias(&x, &prepared).unwrap();
        let loss = prepared.mul(&prepared).unwrap().sum();
        let gradient = loss.vjp_wrt_forward(&[&prepared], &Tensor::scalar(1.)).unwrap().remove(0).unwrap();
        let tangent = current_tangent(&gradient).unwrap().unwrap();
        let outputs = unwrap_values(&[&gradient, &tangent]).unwrap();
        drop(frame);
        drop(level);
        assert!(outputs[0].requires_grad() && outputs[1].requires_grad());
        close(&outputs[0].to_vec(), &[0.4, -0.8]);
        close(&outputs[1].to_vec(), &[0.6, 1.4]);
        for output in &outputs {
            let gradients = output.sum().grad_wrt(&[&x, &base], true).unwrap();
            close(&gradients[0].to_vec(), &[0., 0.]);
            close(&gradients[1].to_vec(), &[0., 0.]);
        }
        if vector_grad { close(&outputs[1].sum().grad_wrt(&[&v], true).unwrap()[0].to_vec(), &[2., 2.]); }
        assert!(x.vjp_wrt(&[&x], &Tensor::ones(&[2]), false).is_err());
    }
}

#[test]
fn sequential_reverse_frames_use_fresh_memos_and_never_replay_callbacks_to_unwrap() {
    use ferro_core::forward_ad::{ReverseFrameGuard, prepare_alias, post_alias, current_tangent, unwrap_values};
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    let x = Tensor::from_vec(vec![0.2, -0.4], &[2]).unwrap();
    let v = Tensor::from_vec(vec![0.3, 0.7], &[2]).unwrap();
    let w = Tensor::scalar(0.5).requires_grad_(true).unwrap();
    let level = DualLevel::functional(&[&x], &[&v]).unwrap();
    let frame = ReverseFrameGuard::new().unwrap();
    let prepared = x.detach().requires_grad_(true).unwrap();
    prepare_alias(&x, &prepared).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let callback_count = count.clone();
    let custom = Tensor::scalar(0.).record_fn_optional(vec![prepared.clone()], move |_| { callback_count.fetch_add(1, Ordering::SeqCst); vec![None] });
    let before_unwrap = count.load(Ordering::SeqCst);
    unwrap_values(&[&custom]).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), before_unwrap);
    let loss = prepared.mul(&prepared).unwrap().mul(&w).unwrap().sum();
    let gradient = loss.vjp_wrt_forward(&[&prepared], &Tensor::scalar(1.)).unwrap().remove(0).unwrap();
    let tangent = current_tangent(&gradient).unwrap().unwrap();
    let first = unwrap_values(&[&gradient, &tangent]).unwrap();
    post_alias(&gradient, &first[0], Some(&first[1])).unwrap();
    drop(frame);
    let frame = ReverseFrameGuard::new().unwrap();
    let prepared = first[0].reshape(&[2]).unwrap();
    prepare_alias(&first[0], &prepared).unwrap();
    let loss = prepared.mul(&prepared).unwrap().sum();
    let gradient = loss.vjp_wrt_forward(&[&prepared], &Tensor::scalar(1.)).unwrap().remove(0).unwrap();
    let tangent = current_tangent(&gradient).unwrap().unwrap();
    let second = unwrap_values(&[&gradient, &tangent]).unwrap();
    drop(frame);
    drop(level);
    close(&second[0].to_vec(), &[0.4, -0.8]);
    close(&second[1].to_vec(), &[0.6, 1.4]);
    let derivative = second[1].sum().grad_wrt(&[&w], true).unwrap().remove(0);
    close(&derivative.to_vec(), &[4.]);
    assert!(!derivative.requires_grad());
    assert_eq!(count.load(Ordering::SeqCst), before_unwrap);
}

#[test]
fn reverse_frame_entry_mode_governs_computed_actual_bases_without_cutting_input_alias_flags() {
    use ferro_core::forward_ad::{ReverseFrameGuard, prepare_alias, current_tangent, unwrap_values};
    for enabled in [false, true] {
        for input_grad in [false, true] {
            let x = Tensor::from_vec(vec![0.2, -0.4], &[2]).unwrap().requires_grad_(input_grad).unwrap();
            let v = leaf(&[0.3, 0.7]);
            let level = DualLevel::functional(&[&x], &[&v]).unwrap();
            let frame = with_grad_enabled(enabled, || ReverseFrameGuard::new()).unwrap();
            let prepared = if input_grad { x.reshape(&[2]).unwrap() } else { x.detach().requires_grad_(true).unwrap() };
            prepare_alias(&x, &prepared).unwrap();
            let base = prepared.mul(&prepared).unwrap();
            let view = with_grad_enabled(false, || base.reshape(&[2]).unwrap());
            let base_direction = current_tangent(&base).unwrap().unwrap();
            let view_direction = current_tangent(&view).unwrap().unwrap();
            let values = unwrap_values(&[&base, &view, &base_direction, &view_direction, &prepared]).unwrap();
            assert_eq!(values[0].requires_grad(), enabled && input_grad);
            assert_eq!(values[1].requires_grad(), enabled && input_grad);
            assert_eq!(values[2].requires_grad(), enabled);
            assert_eq!(values[3].requires_grad(), enabled);
            assert_eq!(values[4].requires_grad(), input_grad);
            assert!(base.requires_grad() && view.requires_grad() && prepared.requires_grad());
            drop(frame);
            drop(level);
            assert!(is_grad_enabled());
        }
    }
}

#[test]
fn auxiliary_no_grad_view_unwrapping_preserves_absent_edges_and_ultimate_base_flags() {
 let mut cases=0;
 for enabled in [false,true] {for xg in [false,true] {for wg in [false,true] {
  let x=Tensor::from_vec(vec![0.2,-0.4],&[2]).unwrap().requires_grad_(xg).unwrap();
  let v=Tensor::from_vec(vec![0.3,0.7],&[2]).unwrap().requires_grad_(true).unwrap();
  let w=Tensor::scalar(0.5).requires_grad_(wg).unwrap();
  let level=DualLevel::functional(&[&x],&[&v]).unwrap();
  let held=with_grad_enabled(enabled,||{let base=x.mul(&x).unwrap().mul(&w).unwrap();with_grad_enabled(false,||base.reshape(&[2]).unwrap())});
  let out=with_grad_enabled(enabled,||level.primal(&held,true)).unwrap();
  assert_eq!(out.requires_grad(),enabled&&(xg||wg));assert!(out.is_leaf());assert_ne!(out.id(),held.id());
  assert_eq!(out._storage_ptr(),held._storage_ptr());assert_eq!(out.strides(),held.strides());assert_eq!(out.storage_offset(),held.storage_offset());
  assert!(level.tangent(&out).unwrap().is_none());
  drop(level);
  if out.requires_grad(){assert!(out.grad_wrt(&[&out],false).is_err());}
  assert!(x.grad().is_none()&&w.grad().is_none()&&v.grad().is_none());cases+=1;
 }}}
 let x=Tensor::from_vec(vec![0.2,-0.4],&[2]).unwrap().requires_grad_(true).unwrap();
 let v=Tensor::from_vec(vec![0.3,0.7],&[2]).unwrap().requires_grad_(true).unwrap();
 let captured=with_grad_enabled(false,||x.reshape(&[2]).unwrap());
 let level=DualLevel::functional(&[&x],&[&v]).unwrap();
 let held=with_grad_enabled(false,||x.reshape(&[2]).unwrap());
 let out=level.primal(&held,true).unwrap();let old=level.primal(&captured,true).unwrap();
 assert_eq!(captured.id(),old.id());assert!(level.tangent(&held).unwrap().is_some());assert!(level.tangent(&out).unwrap().is_none());
 drop(level);x.set_requires_grad(false).unwrap();assert!(!held.requires_grad()&&!out.requires_grad());
 x.set_requires_grad(true).unwrap();assert!(held.requires_grad()&&out.requires_grad());
 assert!(held.grad_wrt(&[&held],false).is_err()&&out.grad_wrt(&[&out],false).is_err());
 assert_eq!(cases, 8);
}
