use ferro_core::optim::{Adam, AdamW, OptimizerState, Sgd};
use ferro_core::{Param, Tensor};

fn state(opt: &impl OptimizerState) -> Vec<(String, Vec<usize>, Vec<u32>)> {
    opt.snapshot().into_iter().map(|(name, t)| (name, t.shape().to_vec(), t.to_vec().iter().map(|v| v.to_bits()).collect())).collect()
}

fn gradient(p: &Param, value: f32) {
    p.tensor().backward_with(&Tensor::scalar(value));
}

#[test]
fn adamw_skipped_and_frozen_slots_match_independent_trajectories() {
    let a = Param::new(Tensor::scalar(1.0));
    let b = Param::new(Tensor::scalar(-0.7));
    let ra = Param::new(a.tensor());
    let rb = Param::new(b.tensor());
    let make = |params| AdamW::new(params, 0.1).with_betas(0.7, 0.9).with_weight_decay(0.2);
    let mut combined = make(vec![a.clone(), b.clone(), a.clone()]);
    let mut left = make(vec![ra.clone()]);
    let mut right = make(vec![rb.clone()]);
    for (ga, gb, frozen) in [
        (Some(1.0), None, false), (Some(-0.5), None, false),
        (None, Some(0.25), false), (Some(0.7), Some(0.8), true),
        (None, None, false), (Some(-0.3), Some(-0.6), false),
    ] {
        b.set_trainable(!frozen);
        rb.set_trainable(!frozen);
        if let Some(g) = ga { gradient(&a, g); gradient(&ra, g); }
        if let Some(g) = gb { gradient(&b, g); gradient(&rb, g); }
        combined.step(); left.step(); right.step();
        assert_eq!(a.tensor().to_vec(), ra.tensor().to_vec());
        assert_eq!(b.tensor().to_vec(), rb.tensor().to_vec());
        let snapshots = [left.snapshot(), right.snapshot()];
        for (name, t) in combined.snapshot() {
            if name == "t" {
                assert_eq!(t.shape(), &[2]);
                let expected: Vec<_> = snapshots.iter().map(|s| s.iter().find(|(n, _)| n == "t").unwrap().1.item()).collect();
                assert_eq!(t.to_vec(), expected);
            } else {
                let (kind, slot) = name.split_once('.').unwrap();
                let expected = &snapshots[slot.parse::<usize>().unwrap()].iter().find(|(n, _)| n == &format!("{kind}.0")).unwrap().1;
                assert_eq!(t.to_vec(), expected.to_vec());
            }
        }
    }
}

fn alternating_isolation<O: OptimizerState>(make: impl Fn(Vec<Param>) -> O, step: impl Fn(&mut O)) {
    let initial = Tensor::scalar(1.0);
    let p = Param::new(initial.clone());
    let tied = p.clone();
    let q = Param::new(initial.clone());
    assert_ne!(p.tensor()._storage_ptr(), q.tensor()._storage_ptr());
    assert_ne!(p.tensor()._storage_ptr(), initial._storage_ptr());
    let mut inactive = make(vec![p.clone(), tied.clone()]);
    let mut active = make(vec![q.clone()]);
    assert_eq!(inactive.state_parameters().unwrap().len(), 1);
    gradient(&p, 1.0);
    step(&mut inactive);
    let before = state(&inactive);
    let identity = (p.tensor().id(), p.tensor()._storage_ptr(), p.tensor()._version(), p.tensor().item());
    gradient(&p, 1.0);
    tied.set_trainable(false);
    assert!(!p.is_trainable());
    assert!(p.tensor().grad().is_none());
    for _ in 0..3 {
        p.tensor().mul(&q.tensor()).unwrap().backward();
        assert!(p.grad().is_none());
        assert!(p.tensor().grad().is_some(), "freeze preserves input differentiation");
        step(&mut active);
        step(&mut inactive);
        assert_eq!(state(&inactive), before);
        assert_eq!((p.tensor().id(), p.tensor()._storage_ptr(), p.tensor()._version(), p.tensor().item()), identity);
    }
    tied.set_trainable(true);
    assert!(p.is_trainable());
    assert!(p.tensor().grad().is_none());
    step(&mut inactive);
    assert_eq!(state(&inactive), before);
    assert_eq!(initial.item(), 1.0);
}

#[test]
fn frozen_optimizers_and_shared_slots_keep_state_isolated() {
    alternating_isolation(|p| Sgd::new(p, 0.1).with_momentum(0.9), |o| o.step());
    alternating_isolation(|p| Adam::new(p, 0.1), |o| o.step());
    alternating_isolation(|p| AdamW::new(p, 0.1), |o| o.step());
}

fn clone_isolation<O: OptimizerState + Clone>(make: impl Fn(Vec<Param>) -> O, step: impl Fn(&mut O)) {
    let p = Param::new(Tensor::scalar(1.0));
    let mut original = make(vec![p.clone()]);
    gradient(&p, 0.5); step(&mut original);
    let mut cloned = original.clone();
    assert_eq!(original.state_parameters().unwrap()[0].identity(), cloned.state_parameters().unwrap()[0].identity());
    for (a, b) in original.state_buffers().unwrap().iter().zip(cloned.state_buffers().unwrap()) {
        assert_ne!(a._storage_ptr(), b._storage_ptr());
        assert_eq!(a.to_vec(), b.to_vec());
    }
    let before = state(&original);
    let p_before = p.tensor().item();
    gradient(&p, -0.25); step(&mut cloned);
    assert_eq!(state(&original), before);
    assert_ne!(p.tensor().item(), p_before, "clones intentionally retain shared parameter slots");
    let before = state(&cloned);
    gradient(&p, 0.75); step(&mut original);
    assert_eq!(state(&cloned), before);
}

#[test]
fn optimizer_clones_own_state_and_share_parameter_slots() {
    clone_isolation(|p| Sgd::new(p, 0.1).with_momentum(0.9), |o| o.step());
    clone_isolation(|p| Adam::new(p, 0.1), |o| o.step());
    clone_isolation(|p| AdamW::new(p, 0.1), |o| o.step());
}

#[test]
fn adamw_counter_roundtrip_and_legacy_scalar_preserve_continuation() {
    let source: Vec<_> = [1.0, 2.0].into_iter().map(|v| Param::new(Tensor::scalar(v))).collect();
    let mut original = AdamW::new(source.clone(), 0.03);
    gradient(&source[0], 0.5); original.step();
    gradient(&source[0], -0.4); gradient(&source[1], 0.7); original.step();
    let checkpoint = original.snapshot();
    let counter = &checkpoint.iter().find(|(n, _)| n == "t").unwrap().1;
    assert_eq!(counter.shape(), &[2]);
    assert_eq!(counter.to_vec(), vec![2.0, 1.0]);
    let target: Vec<_> = source.iter().map(|p| Param::new(p.tensor())).collect();
    let mut restored = AdamW::new(target.clone(), 0.03);
    restored.restore(&checkpoint).unwrap();
    for (index, value) in [(1, -0.2), (0, 0.4), (1, 0.6)] {
        gradient(&source[index], value); gradient(&target[index], value);
        original.step(); restored.step();
        assert_eq!(state(&original), state(&restored));
        for (a, b) in source.iter().zip(&target) { assert_eq!(a.tensor().to_vec(), b.tensor().to_vec()); }
    }
    let mut legacy = checkpoint;
    legacy.iter_mut().find(|(n, _)| n == "t").unwrap().1 = Tensor::scalar(3.0);
    restored.restore(&legacy).unwrap();
    legacy.iter_mut().find(|(n, _)| n == "t").unwrap().1 = Tensor::from_vec(vec![3.0, 3.0], &[2]).unwrap();
    original.restore(&legacy).unwrap();
    assert_eq!(state(&original), state(&restored));
    for p in source.iter().chain(&target) { gradient(p, 0.2); }
    original.step(); restored.step();
    assert_eq!(state(&original), state(&restored));
    for (a, b) in source.iter().zip(&target) { assert_eq!(a.tensor().to_vec(), b.tensor().to_vec()); }
}

#[test]
fn adamw_counter_restore_rejects_invalid_states_atomically() {
    let params = vec![Param::new(Tensor::scalar(1.0)), Param::new(Tensor::scalar(2.0))];
    let mut opt = AdamW::new(params.clone(), 0.1);
    for p in &params { gradient(p, 0.5); }
    opt.step();
    let checkpoint = opt.snapshot();
    let before = state(&opt);
    let buffers = opt.state_buffers().unwrap();
    let versions: Vec<_> = buffers.iter().map(|t| t._version()).collect();
    for invalid in [
        Tensor::from_vec(vec![1.0], &[1]).unwrap(), Tensor::zeros(&[1, 2]), Tensor::zeros(&[0]),
        Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap(), Tensor::from_vec_f64(vec![1.0, 2.0], &[2]).unwrap(),
        Tensor::from_vec(vec![-1.0, 2.0], &[2]).unwrap(), Tensor::from_vec(vec![1.5, 2.0], &[2]).unwrap(),
        Tensor::from_vec(vec![f32::NAN, 2.0], &[2]).unwrap(), Tensor::from_vec(vec![f32::INFINITY, 2.0], &[2]).unwrap(),
        Tensor::from_vec(vec![u32::MAX as f32, 2.0], &[2]).unwrap(), Tensor::scalar(-1.0),
    ] {
        let mut malformed = checkpoint.clone();
        malformed.iter_mut().find(|(n, _)| n == "t").unwrap().1 = invalid;
        assert!(opt.restore(&malformed).is_err());
        assert_eq!(state(&opt), before);
        assert_eq!(buffers.iter().map(|t| t._version()).collect::<Vec<_>>(), versions);
    }
}

#[test]
fn empty_adamw_state_has_an_empty_counter_vector() {
    let mut opt = AdamW::new(vec![], 0.1).capturable();
    assert!(!opt.is_capturable());
    let before = state(&opt);
    opt.step();
    assert_eq!(state(&opt), before);
    let snapshot = opt.snapshot();
    assert_eq!(snapshot.len(), 1);
    assert_eq!(snapshot[0].1.shape(), &[0]);
    opt.restore(&snapshot).unwrap();
    opt.restore(&[("t".into(), Tensor::scalar(3.0))]).unwrap();
    assert_eq!(state(&opt), before);
    assert!(opt.restore(&[("t".into(), Tensor::ones(&[1]))]).is_err());
}
