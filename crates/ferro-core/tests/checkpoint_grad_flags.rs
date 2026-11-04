use ferro_core::{checkpoint::Checkpoint, nn::Module, optim::{OptimizerState, Sgd}, params::Param, rng::Rng, Tensor};

struct Model { left: Param, alias: Param, other: Param }
impl Model {
    fn new(value: f32) -> Self {
        let left = Param::new(Tensor::from_vec(vec![value, value + 0.4], &[2]).unwrap());
        Self { alias: left.clone(), left, other: Param::new(Tensor::from_vec(vec![0.3, -0.2], &[2]).unwrap()) }
    }
    fn optimizer(&self) -> Sgd { Sgd::new(vec![self.left.clone(), self.other.clone()], 0.03).with_momentum(0.6) }
    fn flags(&self, trainable: bool, requires_grad: bool) {
        self.left.set_trainable(trainable);
        self.left.tensor().set_requires_grad(requires_grad).unwrap();
    }
}
impl Module for Model {
    fn forward(&self, x: &Tensor) -> ferro_core::Result<Tensor> {
        x.mul(&self.left.tensor())?.add(&self.alias.tensor())?.add(&self.other.tensor())
    }
    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![("left".into(), self.left.clone()), ("alias".into(), self.alias.clone()), ("other".into(), self.other.clone())]
    }
}

fn advance(model: &Model, optimizer: &mut Sgd, rng: &Rng) -> (Vec<u32>, bool, bool) {
    optimizer.zero_grad();
    let x = Tensor::from_vec((0..2).map(|_| rng.normal()).collect(), &[2]).unwrap();
    let y = model.forward(&x).unwrap();
    let loss = y.mul(&y).unwrap().mean();
    loss.backward();
    let result = (vec![loss.item().to_bits(), x.to_vec()[0].to_bits(), x.to_vec()[1].to_bits()],
        model.left.tensor().grad().is_some(), model.left.grad().is_some());
    optimizer.step();
    optimizer.zero_grad();
    result
}

fn arrays(cp: &Checkpoint) -> Vec<(String, Vec<usize>, Vec<u32>)> {
    let mut result: Vec<_> = cp.tensors.iter().map(|(name,t)| (name.clone(), t.shape().to_vec(), t.to_vec().into_iter().map(f32::to_bits).collect())).collect();
    result.sort_by(|a,b| a.0.cmp(&b.0));
    result
}
fn snapshot(model: &Model, optimizer: &Sgd, rng: &Rng) -> Checkpoint {
    Checkpoint::from_training_state(5, model, &[("sgd", optimizer)], rng).unwrap()
}
fn exact(value: u64) -> Tensor {
    Tensor::from_vec((0..4).map(|i| ((value >> (16 * i)) & 65535) as f32).collect(), &[4]).unwrap()
}
fn replace(cp: &mut Checkpoint, name: &str, value: Tensor) {
    cp.tensors.iter_mut().find(|(key,_)| key == name).unwrap().1 = value;
}

#[test]
fn independent_flags_disk_restore_all_transitions_and_exact_suffix() {
    for source_trainable in [false, true] {
        for source_grad in [false, true] {
            let model = Model::new(0.7); let rng = Rng::new(17); let mut optimizer = model.optimizer();
            for _ in 0..3 { advance(&model, &mut optimizer, &rng); }
            model.flags(source_trainable, source_grad);
            let cp = snapshot(&model, &optimizer, &rng);
            let saved = arrays(&cp);
            let directory = std::env::temp_dir().join(format!("ferro_checkpoint_grad_flags_{}_{}_{}", std::process::id(), source_trainable, source_grad));
            cp.save_to_dir(&directory).unwrap();
            let loaded = Checkpoint::load_from_dir(&directory).unwrap();
            let expected: Vec<_> = (0..4).map(|_| advance(&model, &mut optimizer, &rng)).collect();
            assert!(expected.iter().all(|(_,raw,visible)| *raw == source_grad && *visible == (source_grad && source_trainable)));
            let final_state = arrays(&snapshot(&model, &optimizer, &rng));
            assert_eq!(arrays(&cp), saved);
            for target_trainable in [false, true] {
                for target_grad in [false, true] {
                    let fresh = Model::new(-0.8); let fresh_rng = Rng::new(95); let mut restored = fresh.optimizer();
                    advance(&fresh, &mut restored, &fresh_rng);
                    fresh.flags(target_trainable, target_grad);
                    let handles: Vec<_> = fresh.parameters().iter().map(|p| (p.tensor().id(), p.tensor()._storage_ptr())).collect();
                    loaded.load_training_state_into(&fresh, &mut [("sgd", &mut restored)], &fresh_rng).unwrap();
                    assert_eq!(fresh.left.is_trainable(), source_trainable);
                    assert_eq!(fresh.left.tensor().requires_grad(), source_grad);
                    assert_eq!(fresh.alias.tensor().requires_grad(), source_grad);
                    assert_eq!(fresh.parameters().iter().map(|p| (p.tensor().id(), p.tensor()._storage_ptr())).collect::<Vec<_>>(), handles);
                    assert_eq!(arrays(&snapshot(&fresh, &restored, &fresh_rng)), saved);
                    let actual: Vec<_> = (0..4).map(|_| advance(&fresh, &mut restored, &fresh_rng)).collect();
                    assert_eq!(actual, expected);
                    assert_eq!(arrays(&snapshot(&fresh, &restored, &fresh_rng)), final_state);
                    let expected_rng = Rng::new(0); expected_rng.restore_state(rng.state()).unwrap();
                    assert_eq!(fresh_rng.normal().to_bits(), expected_rng.normal().to_bits());
                }
            }
            std::fs::remove_dir_all(directory).unwrap();
        }
    }
}

#[test]
fn wholly_absent_legacy_namespace_restores_historical_grad_enabled_flags() {
    for trainable in [false, true] {
        let model = Model::new(0.4); let rng = Rng::new(11); let mut optimizer = model.optimizer();
        advance(&model, &mut optimizer, &rng);
        model.flags(trainable, false);
        let mut legacy = snapshot(&model, &optimizer, &rng);
        legacy.tensors.retain(|(name,_)| !name.starts_with("grad_flags."));
        let fresh = Model::new(-1.0); let fresh_rng = Rng::new(99); let mut restored = fresh.optimizer();
        fresh.flags(!trainable, false);
        fresh.other.tensor().set_requires_grad(false).unwrap();
        legacy.load_training_state_into(&fresh, &mut [("sgd", &mut restored)], &fresh_rng).unwrap();
        assert_eq!(fresh.left.is_trainable(), trainable);
        assert!(fresh.parameters().iter().all(|p| p.tensor().requires_grad()));
        model.left.tensor().set_requires_grad(true).unwrap();
        assert_eq!(advance(&fresh, &mut restored, &fresh_rng), advance(&model, &mut optimizer, &rng));
        assert_eq!(arrays(&snapshot(&fresh, &restored, &fresh_rng)), arrays(&snapshot(&model, &optimizer, &rng)));
    }
}

#[test]
fn malformed_partial_tied_and_late_optimizer_state_are_atomic() {
    let model = Model::new(0.7); let rng = Rng::new(17); let mut optimizer = model.optimizer();
    advance(&model, &mut optimizer, &rng); model.flags(false, false);
    let good = snapshot(&model, &optimizer, &rng);
    let mut cases = Vec::new();
    for missing in ["grad_flags.version", "grad_flags.parameter.left", "grad_flags.parameter.alias", "grad_flags.parameter.other"] {
        let mut bad = good.clone(); bad.tensors.retain(|(name,_)| name != missing); cases.push(bad);
    }
    let mut marker_only = good.clone(); marker_only.tensors.retain(|(name,_)| !name.starts_with("grad_flags.parameter.")); cases.push(marker_only);
    for (name,value) in [
        ("grad_flags.version", exact(0)), ("grad_flags.version", exact(2)),
        ("grad_flags.parameter.left", exact(2)), ("grad_flags.parameter.left", Tensor::scalar(0.0)),
        ("grad_flags.parameter.left", Tensor::from_vec_i64(vec![0; 4], &[4]).unwrap()),
        ("grad_flags.parameter.left", Tensor::from_vec(vec![f32::NAN, 0.0, 0.0, 0.0], &[4]).unwrap()),
        ("grad_flags.parameter.left", Tensor::from_vec(vec![0.5, 0.0, 0.0, 0.0], &[4]).unwrap()),
        ("grad_flags.parameter.alias", exact(1)),
        ("optim.sgd.velocity.1", Tensor::zeros(&[3])),
    ] { let mut bad = good.clone(); replace(&mut bad, name, value); cases.push(bad); }
    let mut unknown = good.clone(); unknown.tensors.push(("grad_flags.parameter.unknown".into(), exact(0))); cases.push(unknown);
    let fresh = Model::new(-0.4); let fresh_rng = Rng::new(99); let mut restored = fresh.optimizer();
    advance(&fresh, &mut restored, &fresh_rng);
    for bad in cases {
        let before = arrays(&snapshot(&fresh, &restored, &fresh_rng));
        let handles: Vec<_> = fresh.parameters().iter().map(|p| p.tensor()).chain(restored.state_buffers().unwrap()).collect();
        let identities: Vec<_> = handles.iter().map(|t| (t.id(), t._storage_ptr(), t._version())).collect();
        assert!(bad.load_training_state_into(&fresh, &mut [("sgd", &mut restored)], &fresh_rng).is_err());
        assert_eq!(arrays(&snapshot(&fresh, &restored, &fresh_rng)), before);
        assert_eq!(handles.iter().map(|t| (t.id(), t._storage_ptr(), t._version())).collect::<Vec<_>>(), identities);
    }
}

#[test]
fn shared_storage_with_independent_parameter_leaves_preserves_distinct_grad_flags() {
    let shared = Tensor::ones(&[2]);
    let left = Param::from_shared(&shared, false).unwrap();
    let other = Param::from_shared(&shared, true).unwrap();
    let model = Model { alias: left.clone(), left, other };
    let fresh_shared = Tensor::zeros(&[2]);
    let left = Param::from_shared(&fresh_shared, true).unwrap();
    let other = Param::from_shared(&fresh_shared, false).unwrap();
    let fresh = Model { alias: left.clone(), left, other };
    let rng = Rng::new(3);
    let cp = Checkpoint::from_training_state(0, &model, &[], &rng).unwrap();
    cp.load_training_state_into(&fresh, &mut [], &rng).unwrap();
    assert!(!fresh.left.tensor().requires_grad());
    assert!(fresh.other.tensor().requires_grad());
    assert!(fresh.left.is_trainable() && fresh.other.is_trainable());
    assert_eq!(fresh.left.tensor()._storage_ptr(), fresh.other.tensor()._storage_ptr());
    fresh.forward(&Tensor::ones(&[2])).unwrap().sum().backward();
    assert!(fresh.left.tensor().grad().is_none());
    assert_eq!(fresh.other.tensor().grad().unwrap().to_vec(), vec![1.0, 1.0]);
}
