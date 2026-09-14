use ferro_core::autograd::{is_grad_enabled, set_grad_enabled, with_grad_enabled};
use ferro_core::{Tensor, capture};

fn leaf() -> Tensor { Tensor::scalar(2.0).requires_grad_(true).unwrap() }

#[test]
fn no_grad_suppresses_all_recording_seams_without_changing_inputs() {
    let x = leaf();
    with_grad_enabled(false, || {
        assert!(!is_grad_enabled());
        let custom = Tensor::scalar(3.0).record_fn(vec![x.clone()], |g| vec![g.clone()]);
        for y in [custom, x.mul(&x).unwrap(), x.sum(), x.mean(), x.reshape(&[1]).unwrap()] {
            assert!(!y.requires_grad());
            assert!(y.requires_grad_(true).is_ok(), "no hidden autograd history");
            assert_eq!(y.grad_wrt(&[&x], false).unwrap()[0].item(), 0.0);
        }
        assert!(x.requires_grad());
        assert!(leaf().requires_grad(), "explicit factories are unaffected");
    });
    assert!(x.mul(&x).unwrap().requires_grad());
}

#[test]
fn mode_is_nested_panic_safe_and_thread_local() {
    assert!(is_grad_enabled());
    assert!(set_grad_enabled(false));
    assert!(!set_grad_enabled(true));
    with_grad_enabled(false, || {
        with_grad_enabled(true, || assert!(leaf().sum().requires_grad()));
        assert!(!is_grad_enabled());
        assert!(std::panic::catch_unwind(|| with_grad_enabled(true, || panic!("restore"))).is_err());
        assert!(!is_grad_enabled());
        std::thread::spawn(|| {
            assert!(is_grad_enabled());
            assert!(leaf().sum().requires_grad());
            set_grad_enabled(false);
        }).join().unwrap();
        assert!(!is_grad_enabled());
    });
    assert!(is_grad_enabled());
}

#[test]
fn explicit_capture_remains_independent_of_grad_mode() {
    let x = Tensor::scalar(2.0);
    let y = with_grad_enabled(false, || capture(|| x.mul(&x).unwrap().relu()));
    assert!(!y.requires_grad());
    let compiled = ferro_core::graph::CompiledChain::compile(&y).unwrap();
    x.copy_from(&Tensor::scalar(3.0)).unwrap();
    assert_eq!(compiled.replay().unwrap().item(), 9.0);
}

#[test]
fn create_graph_overrides_no_grad_and_restores_mode_on_error() {
    let x = leaf();
    let y = x.mul(&x).unwrap().mul(&x).unwrap().mean();
    let unsupported = x.relu();
    with_grad_enabled(false, || {
        let g = y.grad_wrt(&[&x], true).unwrap().remove(0);
        assert!(g.requires_grad());
        assert_eq!(g.item(), 12.0);
        assert_eq!(g.grad_wrt(&[&x], true).unwrap()[0].item(), 12.0);
        assert!(!is_grad_enabled());
        assert!(unsupported.grad_wrt(&[&x], true).is_err());
        assert!(!is_grad_enabled());
        assert!(!y.grad_wrt(&[&x], false).unwrap()[0].requires_grad());
    });
    assert!(is_grad_enabled());
}

#[test]
fn existing_graph_backward_works_inside_no_grad() {
    let x = leaf();
    let y = x.mul(&x).unwrap().sum();
    with_grad_enabled(false, || y.backward());
    assert_eq!(x.grad().unwrap().item(), 4.0);
    assert!(is_grad_enabled());
}
