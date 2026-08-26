//! Structural claims for the in-place optimizer steps on the cpu: parameter
//! tensors keep their identity AND storage across steps (the step mutates,
//! never reallocates), storage versions advance so stale graphs fail loudly,
//! the step consumes each updated parameter's grad (the historical
//! replace-the-leaf behavior, kept deliberately - torch's silent cross-step
//! grad accumulation is a footgun), and Param construction takes an owning
//! copy so a step never scribbles over the caller's tensor.

use ferro_core::optim::{Adam, AdamW, Sgd};
use ferro_core::{Param, Tensor};

fn fit_step(p: &Param, target: f32) {
    let t = Tensor::scalar(target);
    let diff = p.tensor().sub(&t.reshape(&[1]).unwrap()).unwrap();
    let loss = diff.mul(&diff).unwrap().sum();
    loss.backward();
}

#[test]
fn params_keep_identity_and_storage_across_steps() {
    let run = |mut step: Box<dyn FnMut()>, p: &Param| {
        let (id0, ptr0, v0) = (
            p.tensor().id(),
            p.tensor()._storage_ptr(),
            p.tensor()._version(),
        );
        for _ in 0..3 {
            fit_step(p, 2.0);
            step();
        }
        assert_eq!(p.tensor().id(), id0, "tensor identity changed");
        assert_eq!(p.tensor()._storage_ptr(), ptr0, "storage reallocated");
        assert_eq!(p.tensor()._version(), v0 + 3, "one version bump per step");
    };

    let p = Param::new(Tensor::from_vec(vec![0.5], &[1]).unwrap());
    let mut o = Sgd::new(vec![p.clone()], 0.1).with_momentum(0.9);
    run(Box::new(move || o.step()), &p);

    let p = Param::new(Tensor::from_vec(vec![0.5], &[1]).unwrap());
    let mut o = Adam::new(vec![p.clone()], 0.1);
    run(Box::new(move || o.step()), &p);

    let p = Param::new(Tensor::from_vec(vec![0.5], &[1]).unwrap());
    let mut o = AdamW::new(vec![p.clone()], 0.1);
    run(Box::new(move || o.step()), &p);
}

#[test]
fn step_consumes_the_grad() {
    let p = Param::new(Tensor::from_vec(vec![1.0], &[1]).unwrap());
    let mut o = Sgd::new(vec![p.clone()], 0.1);
    fit_step(&p, 0.0);
    assert!(p.grad().is_some());
    o.step();
    assert!(
        p.grad().is_none(),
        "an updated param's grad is consumed by the step, matching the old \
         replace-the-leaf behavior"
    );

    // A skipped param (no grad) is untouched and stays skipped.
    let q = Param::new(Tensor::from_vec(vec![3.0], &[1]).unwrap());
    let mut o = Sgd::new(vec![q.clone()], 0.1);
    o.step();
    assert_eq!(q.tensor().to_vec(), vec![3.0]);
    assert!(q.grad().is_none());
}

#[test]
fn param_construction_copies_the_caller_tensor() {
    // The in-place step must never mutate the tensor a Param was built from:
    // construction takes an owning copy. (This is what lets two runs share
    // one init tensor and stay independent.)
    let init = Tensor::from_vec(vec![1.0, -1.0], &[2]).unwrap();
    let p = Param::new(init.clone());
    let mut o = Sgd::new(vec![p.clone()], 0.5);
    fit_step_vec(&p, &[0.0, 0.0]);
    o.step();
    assert_ne!(p.tensor().to_vec(), vec![1.0, -1.0], "step moved the param");
    assert_eq!(init.to_vec(), vec![1.0, -1.0], "caller's tensor mutated");
    assert_eq!(init._version(), 0);
}

fn fit_step_vec(p: &Param, target: &[f32]) {
    let t = Tensor::from_vec(target.to_vec(), &[target.len()]).unwrap();
    let diff = p.tensor().sub(&t).unwrap();
    diff.mul(&diff).unwrap().sum().backward();
}

#[test]
fn stale_graph_backward_after_step_fails_loudly() {
    // Forward -> backward -> step mutates the param; a second backward
    // through the OLD graph must refuse (its saved param version is stale).
    let p = Param::new(Tensor::from_vec(vec![1.0], &[1]).unwrap());
    let mut o = Sgd::new(vec![p.clone()], 0.1);
    let x = Tensor::from_vec(vec![2.0], &[1]).unwrap();
    let loss = p.tensor().mul(&x).unwrap().sum();
    loss.backward();
    o.step();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        loss.backward();
    }));
    assert!(
        panicked.is_err(),
        "backward through a graph whose saved param was stepped must panic"
    );
}

#[test]
fn trajectories_match_the_prewave_reference_values() {
    // Numeric regression anchor: the fused in-place steps promise the exact
    // update-rule expression order of the old allocating implementation.
    // These constants were computed with that formula sequence by hand.
    // SGD momentum: v = 0.9v + g, p -= 0.1v; g = 2(p - 2).
    let p = Param::new(Tensor::from_vec(vec![0.0], &[1]).unwrap());
    let mut o = Sgd::new(vec![p.clone()], 0.1).with_momentum(0.9);
    fit_step(&p, 2.0); // g = -4
    o.step(); // v = -4, p = 0.4
    assert!((p.tensor().to_vec()[0] - 0.4).abs() < 1e-6);
    fit_step(&p, 2.0); // g = 2*(0.4-2) = -3.2
    o.step(); // v = 0.9*-4 + -3.2 = -6.8, p = 0.4 + 0.68 = 1.08
    assert!((p.tensor().to_vec()[0] - 1.08).abs() < 1e-6);
}
