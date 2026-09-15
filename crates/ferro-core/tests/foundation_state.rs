use ferro_core::{checkpoint::Checkpoint, nn::{Linear, Module}, optim::{Adam, OptimizerState, Sgd}, params::Param, rng::Rng, tensor::Tensor};



#[test]
fn corrupt_generation_is_rejected_without_touching_old_generation() {
    let dir = std::env::temp_dir().join(format!("ferro_state_corrupt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Checkpoint::new(1).with_tensor("x", Tensor::scalar(1.0)).save_to_dir(&dir).unwrap();
    let meta = std::fs::read_to_string(dir.join("checkpoint.json")).unwrap();
    let name = meta.lines().find(|l| l.contains("generation")).unwrap().split('"').nth(3).unwrap();
    let file = dir.join(name).join("model.safetensors");
    let mut data = std::fs::read(&file).unwrap();
    *data.last_mut().unwrap() ^= 1;
    std::fs::write(file, data).unwrap();
    assert!(Checkpoint::load_from_dir(&dir).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn owning_clone_does_not_alias_checkpoint() {
    let cp = Checkpoint::new(0).with_tensor("x", Tensor::scalar(1.0));
    let other = cp.clone();
    other.tensor("x").unwrap().fill_(9.0).unwrap();
    assert_eq!(cp.tensor("x").unwrap().item(), 1.0);
}

#[test]
fn named_buffers_restore_is_transactional_and_identity_stable() {
    let a = Tensor::ones(&[2]);
    let b = Tensor::ones(&[3]);
    let cp = Checkpoint::new(0).with_named_state(&[("buffers.mean".into(), Tensor::zeros(&[2])), ("buffers.var".into(), Tensor::zeros(&[4]))]).unwrap();
    assert!(cp.load_named_state_into(&[("buffers.mean".into(), a.clone()), ("buffers.var".into(), b)]).is_err());
    assert_eq!(a.to_vec(), vec![1.0, 1.0]);
}


#[test]
fn conflicting_tied_names_are_rejected_before_mutation() {
    struct Tied(Param);
    impl Module for Tied {
        fn forward(&self, x: &Tensor) -> ferro_core::Result<Tensor> { Ok(x.clone()) }
        fn named_parameters(&self) -> Vec<(String, Param)> { vec![("a".into(), self.0.clone()), ("b".into(), self.0.clone())] }
    }
    let model = Tied(Param::new(Tensor::scalar(1.0)));
    let cp = Checkpoint::new(0).with_tensor("model.a", Tensor::scalar(2.0)).with_tensor("model.b", Tensor::scalar(3.0));
    assert!(cp.load_into_module(&model).is_err());
    assert_eq!(model.0.tensor().item(), 1.0);
}

#[test]
fn successful_restore_preserves_live_parameter_handle() {
    let model = Linear::new(2, 2, &Rng::new(9));
    let live = model.parameters()[0].tensor();
    let mut cp = Checkpoint::from_module(0, &model);
    cp.tensors[0].1 = Tensor::zeros(live.shape());
    cp.load_into_module(&model).unwrap();
    assert_eq!(live.to_vec(), vec![0.0; live.numel()]);
    assert_eq!(live.device(), model.parameters()[0].tensor().device());
}

#[test]
fn rng_state_resumes_next_samples() {
    let rng = Rng::new(77);
    for _ in 0..7 { rng.normal(); }
    let cp = Checkpoint::new(3).with_rng(&rng);
    let expected: Vec<_> = (0..20).map(|_| rng.normal()).collect();
    cp.load_rng_into(&rng).unwrap();
    assert_eq!(expected, (0..20).map(|_| rng.normal()).collect::<Vec<_>>());
}

#[test]
fn optimizer_invalid_late_buffer_is_transactional() {
    let a = Param::new(Tensor::ones(&[2]));
    let b = Param::new(Tensor::ones(&[3]));
    let mut opt = Adam::new(vec![a, b], 0.1);
    let before = opt.snapshot();
    let mut bad = opt.snapshot();
    bad[0].1 = Tensor::ones(&[2]);
    bad[4].1 = Tensor::ones(&[99]);
    assert!(opt.restore(&bad).is_err());
    for ((_, x), (_, y)) in before.iter().zip(opt.snapshot()) { assert_eq!(x.to_vec(), y.to_vec()); }
}

#[test]
fn namespaced_optimizer_configuration_roundtrip() {
    let p = Param::new(Tensor::scalar(1.0));
    let source = Adam::new(vec![p.clone()], 0.037).with_betas(0.8, 0.95).with_eps(0.002);
    let other = Sgd::new(vec![p.clone()], 0.21).with_momentum(0.7);
    let cp = Checkpoint::new(4).with_optimizer("encoder", &source).unwrap().with_optimizer("head", &other).unwrap();
    let mut dst = Adam::new(vec![p.clone()], 1.0);
    let mut dst2 = Sgd::new(vec![p], 1.0);
    cp.load_named_optim_into("encoder", &mut dst).unwrap();
    cp.load_named_optim_into("head", &mut dst2).unwrap();
    assert_eq!(source.configuration(), dst.configuration());
    assert_eq!(other.configuration(), dst2.configuration());
}

#[test]
fn publication_uses_immutable_generation() {
    let dir = std::env::temp_dir().join(format!("ferro_foundation_publish_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    Checkpoint::new(1).with_tensor("x", Tensor::scalar(1.0)).save_to_dir(&dir).unwrap();
    let old = std::fs::read_to_string(dir.join("checkpoint.json")).unwrap();
    let name = old.lines().find(|l| l.contains("generation")).unwrap().split('"').nth(3).unwrap();
    let old_dir = dir.join(name);
    Checkpoint::new(2).with_tensor("x", Tensor::scalar(2.0)).save_to_dir(&dir).unwrap();
    assert_eq!(Checkpoint::load_from_dir(&old_dir).unwrap().step, 1);
    assert_eq!(Checkpoint::load_from_dir(&dir).unwrap().step, 2);
    std::fs::create_dir(dir.join("orphan-interrupted-generation")).unwrap();
    assert_eq!(Checkpoint::load_from_dir(&dir).unwrap().step, 2);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn delayed_snapshot_owns_parameters_and_moments() {
    let model = Linear::new(2, 2, &Rng::new(9));
    let mut opt = Adam::new(model.parameters(), 0.1);
    model.forward(&Tensor::ones(&[1,2])).unwrap().sum().backward();
    opt.step();
    let cp = Checkpoint::from_module_with_optim(1, &model, &opt);
    let before: Vec<_> = cp.tensors.iter().map(|(_, t)| t.to_vec()).collect();
    model.forward(&Tensor::ones(&[1,2])).unwrap().sum().backward();
    opt.step();
    assert_eq!(before, cp.tensors.iter().map(|(_, t)| t.to_vec()).collect::<Vec<_>>());
}

#[test]
fn invalid_model_restore_changes_nothing() {
    let model = Linear::new(2, 2, &Rng::new(9));
    let mut cp = Checkpoint::from_module(1, &model);
    cp.tensors[0].1 = Tensor::zeros(cp.tensors[0].1.shape());
    cp.tensors[1].1 = Tensor::zeros(&[99]);
    let before: Vec<_> = model.parameters().iter().map(|p| p.tensor().to_vec()).collect();
    assert!(cp.load_into_module(&model).is_err());
    assert_eq!(before, model.parameters().iter().map(|p| p.tensor().to_vec()).collect::<Vec<_>>());
}

#[test]
fn tied_parameters_have_one_clipping_contribution_and_slot() {
    let p = Param::new(Tensor::scalar(1.0));
    let mut opt = Sgd::new(vec![p.clone(), p.clone()], 0.1).with_max_grad_norm(1.0);
    p.tensor().backward();
    opt.step();
    assert_eq!(p.tensor().item(), 0.9);
    assert_eq!(opt.snapshot().len(), 1);
}
