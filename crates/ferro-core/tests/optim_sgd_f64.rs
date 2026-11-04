use ferro_core::{Tensor, Param, DType, Result, nn::Module, rng::Rng, checkpoint::Checkpoint,
    optim::{Sgd, Adam, AdamW, Adaptive, Adafactor, Muon, Lbfgs, OptimizerState}};
include!("fixtures/sgd_f64_goldens.rs");

fn tensor(values: &[f64]) -> Tensor { Tensor::from_vec_f64(values.to_vec(), &[values.len()]).unwrap() }
fn parameter(values: &[f64]) -> Param { Param::new(tensor(values)) }
fn bits(value: &Tensor) -> Vec<u64> { value.to_vec_f64().iter().map(|v|v.to_bits()).collect() }
fn gradient(p: &Param, values: &[f64]) { p.tensor().set_leaf_grad(Some(tensor(values))).unwrap(); }
fn encode(state: &[(String,Tensor)]) -> Vec<u8> {
    ferro_core::safetensors::to_safetensors_bytes(&state.iter().map(|(k,v)|(k.as_str(),v)).collect::<Vec<_>>()).unwrap()
}
fn optimizer(p: Vec<Param>) -> Sgd {
    Sgd::new(p,0.).with_f64_options(0.10000000000000002,0.9000000000000001,0.030000000000000002,
        0.10000000000000002,false,false,None).unwrap()
}

#[test]
fn actual_torch_seven_twelve_step_trajectories_match_every_f64_bit() {
    for case in CASES {
        let p=parameter(&[-0.7,0.2,1.1,-0.]);let held=p.tensor();let id=held.id();let storage=held._storage_ptr();
        let c=case.options.map(f64::from_bits);
        let mut opt=Sgd::new(vec![p.clone()],0.).with_f64_options(c[0],c[1],c[2],c[3],case.nesterov,case.maximize,None).unwrap();
        for (step, expected) in case.steps.iter().enumerate() {
            if let Some(values)=expected.gradient { gradient(&p,&values.map(f64::from_bits)); }
            opt.step();assert_eq!(bits(&held),expected.parameter,"step {step}, config {c:?}");
            assert_eq!(p.tensor().id(),id);assert_eq!(p.tensor()._storage_ptr(),storage);assert!(p.grad().is_none());
            if let Some(momentum)=expected.momentum {
                let state=opt.snapshot();assert_eq!(state[0].1.dtype(),DType::F64);assert_eq!(bits(&state[0].1),momentum,"momentum step {step}");
            }
        }
    }
}

#[test]
fn fused_update_and_first_momentum_signed_zero_keep_full_precision() {
    let p=parameter(&[1.]);gradient(&p,&[1.-2f64.powi(-27)]);
    Sgd::new(vec![p.clone()],0.).with_f64_options(1.+2f64.powi(-27),0.,0.,0.,false,false,None).unwrap().step();
    assert_eq!(p.tensor().to_vec_f64(),vec![2f64.powi(-54)]);
    let p=parameter(&[0.,0.]);let mut opt=Sgd::new(vec![p.clone()],0.1).with_momentum(0.9);
    let saved=opt.snapshot();assert_eq!(saved[1].1.item(),0.);let config=opt.configuration();assert_eq!(config.len(),25);
    let mut fresh=Sgd::new(vec![p.clone()],0.);fresh.prepare_cpu_restore(&config,&saved).unwrap()();
    gradient(&p,&[-0.,-0.]);fresh.step();assert_eq!(bits(&fresh.snapshot()[0].1),vec![(-0f64).to_bits();2]);
    gradient(&p,&[1e200, -1e200]);opt= Sgd::new(vec![p.clone()],0.).with_f64_options(1e-200,0.,0.,0.,false,false,None).unwrap();
    opt.step();assert_eq!(p.tensor().to_vec_f64(),vec![-1.,1.]);
}

#[test]
fn shared_storage_leaves_are_sequential_and_have_independent_momentum() {
    let data=tensor(&[0.7,-0.4]);let params:Vec<_>=(0..2).map(|_|Param::from_shared(&data,true).unwrap()).collect();
    let other=tensor(&[0.7,-0.4]);let controls:Vec<_>=(0..2).map(|_|Param::from_shared(&other,true).unwrap()).collect();
    let mut together=optimizer(params.clone());let mut separate:Vec<_>=controls.iter().map(|p|optimizer(vec![p.clone()])).collect();
    for step in 0..8 { for i in 0..2 { if (step+i)%3!=1 { let g=[step as f64*0.13+i as f64,-0.2];gradient(&params[i],&g);gradient(&controls[i],&g); } }
        together.step();for opt in &mut separate {opt.step();}assert_eq!(bits(&data),bits(&other));
        for (i,opt) in separate.iter().enumerate() {assert_eq!(bits(&together.snapshot()[i].1),bits(&opt.snapshot()[0].1));}
    }
}

#[test]
fn typed_restore_validates_late_state_flags_and_exact_chunks_before_mutation() {
    let a=parameter(&[0.7,0.2]);let b=parameter(&[0.1,-0.3]);let mut opt=optimizer(vec![a.clone(),b.clone()]);
    gradient(&a,&[0.3,0.4]);gradient(&b,&[0.5,0.6]);opt.step();let saved=opt.snapshot();let config=opt.configuration();
    let before=encode(&saved);let params=[bits(&a.tensor()),bits(&b.tensor())];
    for value in [-1.,0.5,65536.,f32::NAN,f32::INFINITY] {
        let mut bad=config.clone();bad[20]=value;assert!(opt.prepare_cpu_restore(&bad,&saved).is_err());
        assert_eq!(opt.configuration(),config);assert_eq!(encode(&opt.snapshot()),before);
    }
    for (key,value) in [("velocity.1",Tensor::zeros(&[2])),("initialized.1",Tensor::scalar(0.5)),("velocity.1",tensor(&[0.]))] {
        let mut bad=saved.clone();bad.iter_mut().find(|(k,_)|k==key).unwrap().1=value;
        assert!(opt.prepare_cpu_restore(&config,&bad).is_err());assert_eq!(encode(&opt.snapshot()),before);
        assert_eq!([bits(&a.tensor()),bits(&b.tensor())],params);
    }
    let owned=opt.clone();gradient(&a,&[9.,8.]);opt.step();assert_eq!(encode(&owned.snapshot()),before);
}

struct Model(Param);
impl Module for Model {
    fn forward(&self,x:&Tensor)->Result<Tensor>{x.mul(&self.0.tensor())}
    fn named_parameters(&self)->Vec<(String,Param)>{vec![("weight".into(),self.0.clone())]}
}
#[test]
fn f64_checkpoint_disk_restart_restores_all_state_flags_rng_and_next_updates() {
    let model=Model(parameter(&[0.7,-0.2]));let mut opt=optimizer(model.parameters());let rng=Rng::new(7);
    for _ in 0..4 {gradient(&model.0,&[rng.normal() as f64,rng.normal() as f64]);opt.step();}
    model.0.set_trainable(false);model.0.tensor().set_requires_grad(false).unwrap();
    let saved=Checkpoint::from_training_state(4,&model,&[("sgd",&opt)],&rng).unwrap();
    let path=std::env::temp_dir().join(format!("ferro_sgd_f64_{}",std::process::id()));let _=std::fs::remove_dir_all(&path);
    saved.save_to_dir(&path).unwrap();let loaded=Checkpoint::load_from_dir(&path).unwrap();
    let fresh=Model(parameter(&[8.,9.]));let held=fresh.0.tensor();let id=held.id();let mut other=Sgd::new(fresh.parameters(),0.8);let other_rng=Rng::new(999);
    loaded.load_training_state_into(&fresh,&mut[("sgd",&mut other)],&other_rng).unwrap();
    assert_eq!(held.id(),id);assert!(!fresh.0.is_trainable());assert!(!fresh.0.tensor().requires_grad());
    assert_eq!(encode(&saved.tensors),encode(&Checkpoint::from_training_state(4,&fresh,&[("sgd",&other)],&other_rng).unwrap().tensors));
    model.0.set_trainable(true);fresh.0.set_trainable(true);model.0.tensor().set_requires_grad(true).unwrap();fresh.0.tensor().set_requires_grad(true).unwrap();
    for _ in 0..6 {let g=[rng.normal() as f64,rng.normal() as f64];let h=[other_rng.normal() as f64,other_rng.normal() as f64];assert_eq!(g,h);
        gradient(&model.0,&g);gradient(&fresh.0,&h);opt.step();other.step();assert_eq!(bits(&model.0.tensor()),bits(&held));assert_eq!(encode(&opt.snapshot()),encode(&other.snapshot()));}
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn other_families_reject_f64_at_construction_and_dynamic_replacement() {
    let p=parameter(&[1.]);
    assert_eq!(Adam::new(vec![p.clone()],0.1).snapshot()[0].1.dtype(),DType::F64);
    assert_eq!(AdamW::new(vec![p.clone()],0.1).snapshot()[1].1.dtype(),DType::F64);
    assert_eq!(Adaptive::new(vec![p.clone()],&[5.,0.1,0.9,1e-8,0.,0.,0.,0.]).unwrap().snapshot()[1].1.dtype(),DType::F64);
    assert!(Adafactor::new(vec![p.clone()],&[14.,0.01,-0.8,-1.,0.001,1.,0.,0.]).is_err());
    assert!(Muon::new(vec![p.clone()],&[15.,0.01,0.,0.9,1.,3.4445,-4.775,2.0315,1e-7,5.,0.]).is_err());
    assert!(Lbfgs::new(vec![p],&[13.,0.1,2.,3.,0.,0.,3.]).is_err());
    for adamw in [false,true] {let p=Param::new(Tensor::ones(&[1]));let mut adam=Adam::new(vec![p.clone()],0.1);let mut decoupled=AdamW::new(vec![p.clone()],0.1);
        p.set(tensor(&[1.]));gradient(&p,&[0.1]);let value=bits(&p.tensor());
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||if adamw {decoupled.step()} else {adam.step()}));
        assert!(result.is_err());assert_eq!(bits(&p.tensor()),value);assert!(p.grad().is_some());}
}

#[test]
fn strided_live_gradients_are_read_without_replacing_their_handles() {
    let p=Param::new(Tensor::from_vec_f64(vec![1.;6],&[3,2]).unwrap());
    let seed=Tensor::from_vec_f64(vec![0.1,0.2,0.3,0.4,0.5,0.6],&[2,3]).unwrap();seed.set_requires_grad(true).unwrap();
    let g=seed.transpose(0,1).unwrap();let held=g.clone();let id=g.id();p.tensor().set_leaf_grad(Some(g)).unwrap();
    let mut opt=optimizer(vec![p.clone()]);opt.step();assert_eq!(held.id(),id);assert!(held.requires_grad());
    assert_eq!(held.strides(),&[1,3]);assert_eq!(seed.to_vec_f64(),vec![0.1,0.2,0.3,0.4,0.5,0.6]);assert!(p.grad().is_none());
    assert_eq!(held.sum().grad_wrt(&[&seed],false).unwrap()[0].to_vec_f64(),vec![1.;6]);
}

#[test]
fn mixed_slots_keep_legacy_f32_bits_and_late_rejection_precedes_all_updates() {
    for dampening in [0.,0.3] {
        let a=Param::new(Tensor::from_vec(vec![0.7,-0.3],&[2]).unwrap());let b=parameter(&[0.4,-0.2]);
        let control=Param::new(a.tensor());
        let mut mixed=Sgd::new(vec![a.clone(),b.clone()],0.).with_f64_options(0.017,0.7,0.03,dampening,false,false,None).unwrap();
        let mut legacy=Sgd::new(vec![control.clone()],0.017).with_momentum(0.7).with_weight_decay(0.03).with_dampening(dampening as f32);
        for step in 0..12 {let g=Tensor::from_vec(vec![step as f32*0.03,-0.2],&[2]).unwrap();
            a.tensor().set_leaf_grad(Some(g.clone())).unwrap();control.tensor().set_leaf_grad(Some(g)).unwrap();gradient(&b,&[0.2,0.3]);
            mixed.step();legacy.step();assert_eq!(a.tensor().to_vec(),control.tensor().to_vec());
            assert_eq!(mixed.snapshot()[0].1.to_vec(),legacy.snapshot()[0].1.to_vec());}
        let old=a.tensor().to_vec();let state=encode(&mixed.snapshot());b.set(tensor(&[0.1]));
        a.tensor().set_leaf_grad(Some(Tensor::ones(&[2]))).unwrap();gradient(&b,&[0.1]);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(||mixed.step())).is_err());
        assert_eq!(a.tensor().to_vec(),old);assert_eq!(encode(&mixed.snapshot()),state);assert!(a.grad().is_some());
    }
}
