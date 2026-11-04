use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use ferro_core::Tensor;

fn leaf(value: f32) -> Tensor { Tensor::scalar(value).requires_grad_(true).unwrap() }
fn close(actual: f32, expected: f32) { assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}"); }

#[test]
fn frozen_forward_operands_leave_captured_parameters_and_cotangents_live() {
    let x = leaf(0.3);
    let w = leaf(0.7);
    let cutoff = ferro_core::tensor::tensor_id_watermark();
    let prepared = x.reshape(&[]).unwrap();
    let output = prepared.mul(&w).unwrap().tanh();
    let seed = leaf(0.4);
    let derivative = output.vjp_wrt_frozen(&[&prepared], &seed, cutoff).unwrap().remove(0);
    let tangent = (0.3f32 * 0.7).tanh();
    let slope = 1. - tangent * tangent;
    close(derivative.item(), 0.4 * 0.7 * slope);
    let second = derivative.grad_wrt(&[&x, &w, &seed], false).unwrap();
    close(second[0].item(), 0.);
    close(second[1].item(), 0.4 * slope);
    close(second[2].item(), 0.7 * slope);
    let ordinary = output.vjp_wrt(&[&prepared], &seed, true).unwrap().remove(0);
    let ordinary_second = ordinary.grad_wrt(&[&x], false).unwrap();
    close(ordinary_second[0].item(), -2. * 0.4 * 0.7 * 0.7 * tangent * slope);
    assert!(x.grad().is_none() && w.grad().is_none() && seed.grad().is_none());
}

#[test]
fn frozen_mode_preserves_identity_shared_captures_modes_and_version_checks() {
    let x = leaf(0.3);
    let cutoff = ferro_core::tensor::tensor_id_watermark();
    let prepared = x.reshape(&[]).unwrap();
    let seed = leaf(0.4);
    assert_eq!(prepared.vjp_wrt_frozen(&[&prepared], &seed, cutoff).unwrap()[0].id(), seed.id());
    let duplicates = prepared.add(&prepared).unwrap();
    let duplicate_grad = duplicates.vjp_wrt_frozen(&[&prepared], &seed, cutoff).unwrap().remove(0);
    close(duplicate_grad.grad_wrt(&[&seed], false).unwrap()[0].item(), 2.);
    let captured = prepared.mul(&x).unwrap();
    with_grad_enabled(false, || {
        let derivative = captured.vjp_wrt_frozen(&[&prepared], &seed, cutoff).unwrap().remove(0);
        close(derivative.grad_wrt(&[&x], false).unwrap()[0].item(), 0.4);
        assert!(!is_grad_enabled());
        x._bump_version_for_test();
        assert!(captured.vjp_wrt_frozen(&[&prepared], &seed, cutoff).is_err());
        assert!(!is_grad_enabled());
    });
    assert!(is_grad_enabled());
    assert!(x.grad().is_none() && seed.grad().is_none());
}

#[test]
fn frozen_mode_preserves_direct_primals_but_freezes_computed_views_and_outputs() {
    let x = leaf(0.3);
    let cutoff = ferro_core::tensor::tensor_id_watermark();
    let a = x.reshape(&[]).unwrap();
    let square = a.mul(&a).unwrap();
    let cube = square.mul(&a).unwrap();
    let view = a.reshape(&[]).unwrap();
    let tanh = (0.3f32 * 0.3).tanh();
    let cases = [
        (square, 0.6, 2.0, true),
        (cube, 0.27, 1.2, true),
        (a.powf(3.), 0.27, 1.8, true),
        (view.mul(&view).unwrap(), 0.6, 0., true),
        (a.mul(&a).unwrap().tanh(), 0.6 * (1. - tanh * tanh), 2. * (1. - tanh * tanh), true),
        (a.exp(), 0.3f32.exp(), 0., false),
        (a.tanh(), 1. - 0.3f32.tanh().powi(2), 0., false),
        (a.sigmoid(), { let y = 1. / (1. + (-0.3f32).exp()); y * (1. - y) }, 0., false),
        (a.sqrt(), 0.5 / 0.3f32.sqrt(), 0., false),
        (a.relu(), 1., 0., false),
        (a.log(), 1. / 0.3, -1. / (0.3 * 0.3), true),
    ];
    for (output, slope, curvature, public_grad_flag) in cases {
        for live_seed in [false, true] {
            let seed = Tensor::scalar(0.4).requires_grad_(live_seed).unwrap();
            let derivative = output.vjp_wrt_frozen(&[&a], &seed, cutoff).unwrap().remove(0);
            close(derivative.item(), 0.4 * slope);
            assert_eq!(derivative.requires_grad(), public_grad_flag || live_seed);
            let second = derivative.grad_wrt(&[&x, &seed], false).unwrap();
            close(second[0].item(), 0.4 * curvature);
            close(second[1].item(), if live_seed { slope } else { 0. });
        }
    }
    assert!(x.grad().is_none());
}

#[test]
fn frozen_views_distinguish_real_primal_flags_synthetic_inputs_and_computed_storage() {
    let x = leaf(0.3);
    let captured = leaf(0.7);
    let cutoff = ferro_core::tensor::tensor_id_watermark();
    let prepared = x.reshape(&[]).unwrap();
    let view = prepared.reshape(&[]).unwrap();
    let output = view.mul(&view).unwrap();
    let seed = Tensor::scalar(0.4);
    let real = output.vjp_wrt_frozen(&[&prepared], &seed, cutoff).unwrap().remove(0);
    let synthetic = output.vjp_wrt_frozen_excluding(&[&prepared], &seed, cutoff, &[&prepared]).unwrap().remove(0);
    close(real.item(), 0.24);
    close(synthetic.item(), 0.24);
    assert!(real.requires_grad());
    assert!(!synthetic.requires_grad());
    close(real.grad_wrt(&[&x], false).unwrap()[0].item(), 0.);

    let computed_view = prepared.mul(&prepared).unwrap().reshape(&[]).unwrap();
    let computed_output = computed_view.mul(&computed_view).unwrap();
    let computed = computed_output.vjp_wrt_frozen(&[&prepared], &seed, cutoff).unwrap().remove(0);
    close(computed.item(), 4. * 0.3f32.powi(3) * 0.4);
    close(computed.grad_wrt(&[&x], false).unwrap()[0].item(), 4. * 0.3f32.powi(2) * 0.4);

    let captured_view = captured.reshape(&[]).unwrap();
    let captured_output = prepared.mul(&captured_view).unwrap();
    let gradient = captured_output.vjp_wrt_frozen_excluding(&[&prepared], &seed, cutoff, &[&prepared]).unwrap().remove(0);
    close(gradient.item(), 0.28);
    assert!(gradient.requires_grad());
    close(gradient.grad_wrt(&[&captured], false).unwrap()[0].item(), 0.);

    with_grad_enabled(false, || {
        assert!(output.vjp_wrt_frozen_excluding(&[&prepared], &seed, cutoff, &[&captured]).is_err());
        assert!(!is_grad_enabled());
    });
    assert!(is_grad_enabled());
    assert!(x.grad().is_none() && captured.grad().is_none() && prepared.grad().is_none());
}

#[test]
fn frozen_saved_output_versions_protect_later_pullbacks() {
    let x = leaf(0.3);
    let cutoff = ferro_core::tensor::tensor_id_watermark();
    let a = x.reshape(&[]).unwrap();
    let output = a.tanh();
    let seed = leaf(0.4);
    let derivative = output.vjp_wrt_frozen(&[&a], &seed, cutoff).unwrap().remove(0);
    output.detach().copy_from(&Tensor::scalar(0.9)).unwrap();
    assert!(derivative.grad_wrt(&[&seed], false).is_err());
    assert!(output.vjp_wrt_frozen(&[&a], &seed, cutoff).is_err());
    assert!(x.grad().is_none() && seed.grad().is_none() && derivative.grad().is_none());
    assert!(is_grad_enabled());
}
