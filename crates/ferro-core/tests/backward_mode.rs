use std::sync::{Arc,atomic::{AtomicBool,Ordering}};
use ferro_core::autograd::{is_grad_enabled,with_grad_enabled};
use ferro_core::Tensor;

#[test]
fn ordinary_callbacks_compute_without_reverse_recording_and_restore_the_callers_mode() {
    for enabled in [false,true] {
        let x=Tensor::from_vec(vec![2.,3.],&[2]).unwrap().requires_grad_(true).unwrap();
        let parent=x.clone();let seen=Arc::new(AtomicBool::new(true));let capture=seen.clone();
        let y=Tensor::from_vec(vec![4.,9.],&[2]).unwrap().record_fn(vec![x.clone()],move|g|{
            capture.store(is_grad_enabled(),Ordering::SeqCst);vec![g.mul(&parent).unwrap()]
        });
        with_grad_enabled(enabled,||{
            y.backward_with(&Tensor::ones(&[2]));
            assert_eq!(is_grad_enabled(),enabled);
        });
        assert!(!seen.load(Ordering::SeqCst));
        let gradient=x.grad().unwrap();assert!(!gradient.requires_grad()&&gradient.is_leaf());assert_eq!(gradient.to_vec(),vec![2.,3.]);
    }
    assert!(is_grad_enabled());
}

#[test]
fn callback_panics_restore_mode_and_leave_preexisting_slots_unpublished() {
    for enabled in [false,true] {
        let x=Tensor::scalar(2.).requires_grad_(true).unwrap();
        x.set_leaf_grad(Some(Tensor::scalar(7.))).unwrap();let held=x.grad().unwrap();
        let y=Tensor::scalar(4.).record_fn(vec![x.clone()],|_|{
            assert!(!is_grad_enabled());panic!("deliberate callback failure")
        });
        with_grad_enabled(enabled,||{
            let failed=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||y.backward()));
            assert!(failed.is_err());assert_eq!(is_grad_enabled(),enabled);
        });
        assert_eq!(x.grad().unwrap().id(),held.id());assert_eq!(held.to_vec(),vec![7.]);
    }
}
