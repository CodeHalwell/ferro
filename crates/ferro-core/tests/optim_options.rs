use ferro_core::{checkpoint::Checkpoint, nn::Module, optim::{Adam, AdamW, OptimizerState, Sgd}, rng::Rng, Param, Result, Tensor};

struct Model { parameter: Param }
impl Model { fn new(value: f32) -> Self { Self { parameter: Param::new(Tensor::full(&[3],value)) } } }
impl Module for Model {
    fn forward(&self, x: &Tensor) -> Result<Tensor> { self.parameter.tensor().mul(x) }
    fn named_parameters(&self) -> Vec<(String,Param)> { vec![("weight".into(),self.parameter.clone())] }
}
fn gradient(parameter: &Param, values: &[f32]) { parameter.tensor().backward_with(&Tensor::from_vec(values.to_vec(),parameter.tensor().shape()).unwrap()); }
fn state(optimizer: &dyn OptimizerState) -> Vec<(String,Vec<usize>,Vec<u32>)> {
    optimizer.snapshot().into_iter().map(|(name,t)| (name,t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect()
}
fn checkpoint(model: &Model, optimizer: &dyn OptimizerState, rng: &Rng) -> Checkpoint {
    Checkpoint::from_training_state(3,model,&[("main",optimizer)],rng).unwrap()
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(),expected.len());
    for (a,b) in actual.iter().zip(expected) { assert!((a-b).abs() < 3e-6, "{actual:?} != {expected:?}"); }
}

#[test]
fn coupled_decay_dampening_amsgrad_and_maximize_match_torch_history() {
    // Actual Torch 2.13 CPU foreach=False, f32-rounded scalar settings.
    for adam in [false,true] {
        let p = Param::new(Tensor::from_vec(vec![0.75,-1.2,0.4],&[3]).unwrap());
        let mut sgd = Sgd::new(vec![p.clone()],0.03).with_momentum(0.7).with_dampening(0.3).with_weight_decay(0.07).with_maximize(true);
        let mut adaptive = Adam::new(vec![p.clone()],0.02).with_betas(0.7,0.8).with_eps(0.001).with_weight_decay(0.03).with_amsgrad(true).with_maximize(true);
        let pointer = p.tensor()._storage_ptr();
        for step in 0..20 {
            if step==10 { sgd.set_lr(0.03*0.6); adaptive.set_lr(0.02*0.6); }
            if step%6!=2 { gradient(&p,&[(step%7) as f32*0.125-0.375,0.25/(step+1) as f32,if step%2==0 {-0.375} else {0.625}]); }
            if adam { adaptive.step(); } else { sgd.step(); }
            assert!(p.grad().is_none());
        }
        assert_eq!(p.tensor()._storage_ptr(),pointer);
        let arrays = if adam { adaptive.snapshot() } else { sgd.snapshot() };
        if adam {
            close(&p.tensor().to_vec(),&[0.70953125,-1.0400872,0.4768587]);
            for (name,values) in [("m.0",vec![-0.07829767,-0.046919607,-0.23438601]),("v.0",vec![0.04025575,0.0030373738,0.28023273]),("vmax.0",vec![0.07246082,0.018233126,0.2883338])] {
                close(&arrays.iter().find(|(n,_)|n==name).unwrap().1.to_vec(),&values);
            }
        } else {
            close(&p.tensor().to_vec(),&[0.6971666,-1.0822604,0.52579576]);
            close(&arrays.iter().find(|(n,_)|n=="velocity.0").unwrap().1.to_vec(),&[-0.11793361,-0.21471946,-0.49711215]);
            assert_eq!(arrays.iter().find(|(n,_)|n=="initialized.0").unwrap().1.item(),1.);
        }
    }
}

#[test]
fn saved_configuration_selects_exact_legacy_or_extended_optimizer_state_schema() {
    for extended in [false,true] {
        let model = Model::new(0.7); let rng = Rng::new(7);
        let mut source = Sgd::new(model.parameters(),0.03).with_momentum(0.7).with_dampening(if extended {0.3} else {0.});
        gradient(&model.parameter,&[0.2,-0.4,0.6]); source.step();
        let saved = checkpoint(&model,&source,&rng);
        let target = Model::new(9.); let other_rng = Rng::new(99);
        let mut destination = Sgd::new(target.parameters(),0.8).with_momentum(0.2).with_dampening(if extended {0.} else {0.4});
        saved.load_training_state_into(&target,&mut [("main",&mut destination)],&other_rng).unwrap();
        assert_eq!(source.configuration(),destination.configuration());
        for p in [&model.parameter,&target.parameter] { gradient(p,&[-0.3,0.7,-0.2]); }
        source.step(); destination.step();
        assert_eq!(model.parameter.tensor().to_vec(),target.parameter.tensor().to_vec());
        assert_eq!(state(&source),state(&destination)); assert_eq!(rng.state(),other_rng.state());

        let model = Model::new(0.7); let rng = Rng::new(7);
        let mut source = Adam::new(model.parameters(),0.03).with_amsgrad(extended);
        gradient(&model.parameter,&[0.2,-0.4,0.6]); source.step();
        let saved = checkpoint(&model,&source,&rng);
        let target = Model::new(9.); let other_rng = Rng::new(99);
        let mut destination = Adam::new(target.parameters(),0.8).with_amsgrad(!extended);
        gradient(&target.parameter,&[0.8,0.2,-0.7]); destination.step();
        saved.load_training_state_into(&target,&mut [("main",&mut destination)],&other_rng).unwrap();
        assert_eq!(source.configuration(),destination.configuration());
        for p in [&model.parameter,&target.parameter] { gradient(p,&[-0.3,0.7,-0.2]); }
        source.step(); destination.step();
        assert_eq!(model.parameter.tensor().to_vec(),target.parameter.tensor().to_vec());
        assert_eq!(state(&source),state(&destination)); assert_eq!(rng.state(),other_rng.state());
    }
}

#[test]
fn unknown_keys_bad_namespaces_and_config_state_mismatch_reject_before_mutation() {
    let model = Model::new(0.7); let rng = Rng::new(7);
    let mut source = Adam::new(model.parameters(),0.03).with_amsgrad(true);
    gradient(&model.parameter,&[0.2,-0.4,0.6]); source.step();
    let saved = checkpoint(&model,&source,&rng);
    let target = Model::new(9.); let other_rng = Rng::new(99);
    let mut destination = Adam::new(target.parameters(),0.8);
    for fault in 0..4 {
        let mut malformed = saved.clone();
        match fault {
            0 => malformed.tensors.push(("optim.main.unexpected".into(),Tensor::scalar(1.))),
            1 => malformed.tensors.push(("optim.other.t.0".into(),Tensor::scalar(1.))),
            2 => malformed.tensors.retain(|(name,_)|name!="optim.main.vmax.0"),
            _ => malformed.tensors.iter_mut().find(|(name,_)|name=="optim.main.config").unwrap().1 = Tensor::from_vec(vec![2.,0.03,0.9,0.999,1e-8,-1.],&[6]).unwrap(),
        }
        let before = (target.parameter.tensor().to_vec(),state(&destination),destination.configuration(),other_rng.state(),target.parameter.tensor()._version());
        assert!(malformed.load_training_state_into(&target,&mut [("main",&mut destination)],&other_rng).is_err());
        assert_eq!((target.parameter.tensor().to_vec(),state(&destination),destination.configuration(),other_rng.state(),target.parameter.tensor()._version()),before);
    }
}

struct Custom { parameter: Param }
impl OptimizerState for Custom {
    fn snapshot(&self) -> Vec<(String,Tensor)> { vec![("known".into(),Tensor::scalar(1.))] }
    fn state_parameters(&self) -> Option<Vec<Param>> { Some(vec![self.parameter.clone()]) }
    fn state_buffers(&self) -> Option<Vec<Tensor>> { Some(vec![]) }
    fn prepare_cpu_restore<'a>(&'a mut self, _: &[f32], _: &[(String,Tensor)]) -> Result<Box<dyn FnOnce()+'a>> { Ok(Box::new(||{})) }
    fn restore(&mut self, _: &[(String,Tensor)]) -> Result<()> { Ok(()) }
}

#[test]
fn custom_optimizer_default_retains_strict_snapshot_keys() {
    let model = Model::new(0.7); let rng = Rng::new(7);
    let source = Custom { parameter: model.parameter.clone() };
    let mut saved = checkpoint(&model,&source,&rng);
    saved.tensors.push(("optim.main.unexpected".into(),Tensor::scalar(1.)));
    let target = Model::new(9.); let other_rng = Rng::new(99);
    let mut destination = Custom { parameter: target.parameter.clone() };
    assert!(saved.load_training_state_into(&target,&mut [("main",&mut destination)],&other_rng).is_err());
    assert_eq!(target.parameter.tensor().to_vec(),vec![9.;3]); assert_eq!(other_rng.state(),Rng::new(99).state());
}

#[test]
fn adamw_amsgrad_maximize_matches_torch_and_restores_both_schema_directions() {
    let p = Param::new(Tensor::from_vec(vec![0.75,-1.2,0.4],&[3]).unwrap());
    let mut optimizer = AdamW::new(vec![p.clone()],0.02).with_betas(0.7,0.8).with_weight_decay(0.03).with_amsgrad(true).with_maximize(true);
    let pointer = p.tensor()._storage_ptr();
    for step in 0..20 {
        if step==10 { optimizer.set_lr(0.02*0.6); }
        if step%6!=2 { gradient(&p,&[(step%7) as f32*0.125-0.375,0.25/(step+1) as f32,if step%2==0 {-0.375} else {0.625}]); }
        optimizer.step();
    }
    // Actual Torch CPU foreach=False, with f32-rounded scalar settings.
    close(&p.tensor().to_vec(),&[0.72025299,-1.06787193,0.47817069]);
    let arrays = optimizer.snapshot();
    for (name,values) in [("m.0",vec![-0.099404715,-0.015332542,-0.24822132]),("v.0",vec![0.042985655,0.0008087763,0.28628197]),("vmax.0",vec![0.07169379,0.013125,0.29603752])] {
        close(&arrays.iter().find(|(n,_)|n==name).unwrap().1.to_vec(),&values);
    }
    assert_eq!(p.tensor()._storage_ptr(),pointer);
    assert_eq!(arrays[0].1.to_vec(),vec![17.]);
    let clone = optimizer.clone();
    gradient(&p,&[0.1,0.2,0.3]); optimizer.step();
    assert_eq!(state(&clone), arrays.iter().map(|(n,t)| (n.clone(),t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect::<Vec<_>>());
    for extended in [false,true] {
        let model = Model::new(0.7); let rng = Rng::new(7);
        let mut source = AdamW::new(model.parameters(),0.03).with_amsgrad(extended).with_maximize(extended);
        gradient(&model.parameter,&[0.2,-0.4,0.6]); source.step();
        let saved = checkpoint(&model,&source,&rng);
        let target = Model::new(9.); let other_rng = Rng::new(99);
        let mut destination = AdamW::new(target.parameters(),0.8).with_amsgrad(!extended);
        gradient(&target.parameter,&[0.8,0.2,-0.7]); destination.step();
        saved.load_training_state_into(&target,&mut [("main",&mut destination)],&other_rng).unwrap();
        assert_eq!(source.configuration(),destination.configuration());
        for p in [&model.parameter,&target.parameter] { gradient(p,&[-0.3,0.7,-0.2]); }
        source.step(); destination.step();
        assert_eq!(model.parameter.tensor().to_vec(),target.parameter.tensor().to_vec());
        assert_eq!(state(&source),state(&destination)); assert_eq!(rng.state(),other_rng.state());
        let mut malformed = checkpoint(&model,&source,&rng);
        malformed.tensors.push(("optim.main.unexpected".into(),Tensor::scalar(1.)));
        let before = (target.parameter.tensor().to_vec(),state(&destination),destination.configuration(),other_rng.state());
        assert!(malformed.load_training_state_into(&target,&mut [("main",&mut destination)],&other_rng).is_err());
        assert_eq!((target.parameter.tensor().to_vec(),state(&destination),destination.configuration(),other_rng.state()),before);
    }
}
