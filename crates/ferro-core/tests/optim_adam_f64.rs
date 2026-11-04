use ferro_core::{Tensor, Param, DType, optim::{Adam,AdamW,OptimizerState}};
include!("fixtures/adam_f64_goldens.rs");
fn tensor(v:&[f64])->Tensor { Tensor::from_vec_f64(v.to_vec(),&[v.len()]).unwrap() }
fn parameter(v:&[f64])->Param { Param::new(tensor(v)) }
fn grad(p:&Param,v:&[f64]) { p.tensor().set_leaf_grad(Some(tensor(v))).unwrap(); }
fn bytes(state:&[(String,Tensor)])->Vec<u8> { ferro_core::safetensors::to_safetensors_bytes(&state.iter().map(|(k,v)|(k.as_str(),v)).collect::<Vec<_>>()).unwrap() }
fn exact(actual:&Tensor,expected:&[u64],label:&str) {
    for (i,(x,&bits)) in actual.to_vec_f64().iter().zip(expected).enumerate() {
        let y=f64::from_bits(bits);assert!(x.to_bits()==bits || x.is_nan()&&y.is_nan(),"{label}[{i}]: {x:?}({:x}) != {y:?}({bits:x})",x.to_bits());
    }
}
#[test]
fn actual_torch_eighty_double_configurations_match_parameters_and_all_moments() {
 for (ci,c) in GOLDENS.iter().enumerate() {let p=parameter(&[0.7,-0.3,1e-200,1e150]);let held=p.tensor();let id=held.id();let ptr=held._storage_ptr();
  let [b1,b2,lr,eps]=c.config.map(f64::from_bits);
  let mut a=Adam::new(vec![p.clone()],0.).with_f64_options(lr,b1,b2,eps,0.1234567891234567,c.amsgrad,c.maximize,None).unwrap();
  let mut w=AdamW::new(vec![p.clone()],0.).with_f64_options(lr,b1,b2,eps,0.1234567891234567,c.amsgrad,c.maximize,None).unwrap();
  for (si,g) in [[0.2,-0.4,1e-200,1e100],[-0.7,0.1,-1e-150,1e120]].iter().enumerate() {grad(&p,g);if c.adamw {w.step()}else{a.step()}
   let state=if c.adamw{w.snapshot()}else{a.snapshot()};exact(&held,&c.updates[si][0],&format!("case{ci} step{si} p"));
   for (j,key) in ["m.0","v.0","vmax.0"].iter().enumerate() {if j==2&&!c.amsgrad{continue}let t=&state.iter().find(|(k,_)|k==key).unwrap().1;assert_eq!(t.dtype(),DType::F64);exact(t,&c.updates[si][j+1],&format!("case{ci} step{si} {key}"));}
   assert_eq!(p.tensor().id(),id);assert_eq!(p.tensor()._storage_ptr(),ptr);assert!(p.grad().is_none());
  }
 }
}
#[test]
fn precise_configuration_and_typed_restore_reject_late_invalid_state_atomically() {
 for adamw in [false,true] {let p=parameter(&[0.7,-0.2]);let q=parameter(&[0.1,0.3]);
  let mut a=Adam::new(vec![p.clone(),q.clone()],0.).with_f64_options(0.1234567891234567,0.9123456789123456,0.9999999999999999,1e-200,0.01,true,false,None).unwrap();
  let mut w=AdamW::new(vec![p.clone(),q.clone()],0.).with_f64_options(0.1234567891234567,0.9123456789123456,0.9999999999999999,1e-200,0.01,true,false,None).unwrap();
  grad(&p,&[0.2,0.3]);grad(&q,&[0.4,0.5]);if adamw{w.step()}else{a.step()}
  let opt:&mut dyn OptimizerState=if adamw{&mut w}else{&mut a};let config=opt.configuration();assert_eq!(config.len(),29);let saved=opt.snapshot();let before=bytes(&saved);
  for index in [5,17,28] {let mut bad=config.clone();bad[index]=65536.;assert!(opt.prepare_cpu_restore(&bad,&saved).is_err());assert_eq!(opt.configuration(),config);assert_eq!(bytes(&opt.snapshot()),before);}
  for value in [Tensor::zeros(&[2]),tensor(&[0.])] {let mut bad=saved.clone();bad.iter_mut().find(|(k,_)|k=="v.1").unwrap().1=value;assert!(opt.prepare_cpu_restore(&config,&bad).is_err());assert_eq!(bytes(&opt.snapshot()),before);}
  opt.prepare_cpu_restore(&config,&saved).unwrap()();assert_eq!(bytes(&opt.snapshot()),before);
 }
}

fn configured(params:Vec<Param>,adamw:bool)->Box<dyn OptimizerState> {
 if adamw {Box::new(AdamW::new(params,0.).with_f64_options(0.017,0.7,0.93,1e-8,0.03,true,false,None).unwrap())}
 else {Box::new(Adam::new(params,0.).with_f64_options(0.017,0.7,0.93,1e-8,0.03,true,false,None).unwrap())}
}
#[test]
fn mixed_slots_preserve_legacy_float32_arithmetic_and_late_failure_is_atomic() {
 for adamw in [false,true] {
  let a=Param::new(Tensor::from_vec(vec![0.7,-0.3],&[2]).unwrap());let b=parameter(&[0.4,-0.2]);let control=Param::new(a.tensor());
  let mut ma=Adam::new(vec![a.clone(),b.clone()],0.).with_f64_options(0.017,0.7,0.93,1e-8,0.03,true,false,None).unwrap();
  let mut mw=AdamW::new(vec![a.clone(),b.clone()],0.).with_f64_options(0.017,0.7,0.93,1e-8,0.03,true,false,None).unwrap();
  let mut ca=Adam::new(vec![control.clone()],0.017).with_betas(0.7,0.93).with_weight_decay(0.03).with_amsgrad(true);
  let mut cw=AdamW::new(vec![control.clone()],0.017).with_betas(0.7,0.93).with_weight_decay(0.03).with_amsgrad(true);
  for step in 0..7 {let g=Tensor::from_vec(vec![step as f32*0.03,-0.2],&[2]).unwrap();a.tensor().set_leaf_grad(Some(g.clone())).unwrap();control.tensor().set_leaf_grad(Some(g)).unwrap();grad(&b,&[0.2,0.3]);
   if adamw {mw.step();cw.step()}else{ma.step();ca.step()}
   assert_eq!(a.tensor().to_vec(),control.tensor().to_vec());
   let mixed=if adamw{mw.snapshot()}else{ma.snapshot()};let legacy=if adamw{cw.snapshot()}else{ca.snapshot()};
   for key in ["m.0","v.0","vmax.0"] {assert_eq!(mixed.iter().find(|(k,_)|k==key).unwrap().1.to_vec(),legacy.iter().find(|(k,_)|k==key).unwrap().1.to_vec());}
  }
  let before=a.tensor().to_vec();let state=bytes(&if adamw{mw.snapshot()}else{ma.snapshot()});b.set(tensor(&[0.1]));a.tensor().set_leaf_grad(Some(Tensor::ones(&[2]))).unwrap();grad(&b,&[0.1]);
  let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||if adamw{mw.step()}else{ma.step()}));assert!(result.is_err());assert_eq!(a.tensor().to_vec(),before);assert_eq!(bytes(&if adamw{mw.snapshot()}else{ma.snapshot()}),state);assert!(a.grad().is_some());
 }
}
#[test]
fn shared_storage_leaves_advance_sequentially_with_independent_state() {
 for adamw in [false,true] {let data=tensor(&[0.7,-0.4]);let control=tensor(&[0.7,-0.4]);let ps:Vec<_>=(0..2).map(|_|Param::from_shared(&data,true).unwrap()).collect();let cs:Vec<_>=(0..2).map(|_|Param::from_shared(&control,true).unwrap()).collect();
  let mut ma=Adam::new(ps.clone(),0.01).with_amsgrad(true);let mut mw=AdamW::new(ps.clone(),0.01).with_amsgrad(true);
  let mut ca:Vec<_>=cs.iter().map(|p|Adam::new(vec![p.clone()],0.01).with_amsgrad(true)).collect();let mut cw:Vec<_>=cs.iter().map(|p|AdamW::new(vec![p.clone()],0.01).with_amsgrad(true)).collect();
  for step in 0..8 {for i in 0..2 {if (step+i)%3!=1 {let g=[step as f64*0.13+i as f64,-0.2];grad(&ps[i],&g);grad(&cs[i],&g);}}
   if adamw{mw.step();for o in &mut cw{o.step()}}else{ma.step();for o in &mut ca{o.step()}}
   assert_eq!(data.to_vec_f64(),control.to_vec_f64());
   for i in 0..2 {let mixed=if adamw{mw.snapshot()}else{ma.snapshot()};let single=if adamw{cw[i].snapshot()}else{ca[i].snapshot()};for key in ["m","v","vmax"] {assert_eq!(mixed.iter().find(|(k,_)|k==&format!("{key}.{i}")).unwrap().1.to_vec_f64(),single.iter().find(|(k,_)|k==&format!("{key}.0")).unwrap().1.to_vec_f64());}}
  }
 }
}
#[test]
fn exact_counter_state_and_scalar_codec_validate_before_configuration_mutation() {
 for adamw in [false,true] {let p=parameter(&[0.7,-0.2]);let mut opt=configured(vec![p.clone()],adamw);let config=opt.configuration();let before=bytes(&opt.snapshot());
  for chunk in [f32::NAN,-1.,65536.,0.5] {let mut bad=config.clone();bad[10]=chunk;assert!(opt.restore_configuration(&bad).is_err());assert_eq!(opt.configuration(),config);assert_eq!(bytes(&opt.snapshot()),before);}
  let mut state=opt.snapshot();let counter=if adamw{"t"}else{"t.0"};let shape=if adamw{vec![1]}else{vec![]};state.iter_mut().find(|(k,_)|k==counter).unwrap().1=Tensor::from_vec_f64(vec![u32::MAX as f64],&shape).unwrap();
  opt.prepare_cpu_restore(&config,&state).unwrap()();assert_eq!(opt.snapshot().iter().find(|(k,_)|k==counter).unwrap().1.to_vec_f64(),vec![u32::MAX as f64]);
  let fresh=parameter(&[0.]);assert!(Adam::new(vec![fresh],0.).with_f64_options(0.1,f64::NAN,0.9,1e-8,0.,false,false,None).is_err());
 }
 let p=Param::new(Tensor::ones(&[1]));let q=parameter(&[1.]);assert!(Adam::new(vec![p,q],0.).with_f64_options(1e200,0.9,0.999,1e-8,0.,false,false,None).is_err());
}
#[test]
fn strided_live_gradients_keep_their_history_and_storage_untouched() {
 for adamw in [false,true] {let p=Param::new(Tensor::from_vec_f64(vec![1.;6],&[3,2]).unwrap());let seed=Tensor::from_vec_f64(vec![0.1,0.2,0.3,0.4,0.5,0.6],&[2,3]).unwrap();seed.set_requires_grad(true).unwrap();let g=seed.transpose(0,1).unwrap();let id=g.id();let ptr=g._storage_ptr();p.tensor().set_leaf_grad(Some(g.clone())).unwrap();
  if adamw{AdamW::new(vec![p.clone()],0.01).step()}else{Adam::new(vec![p.clone()],0.01).step()}
  assert!(p.grad().is_none());assert_eq!(g.id(),id);assert_eq!(g._storage_ptr(),ptr);assert!(g.requires_grad());assert_eq!(g.strides(),&[1,3]);assert_eq!(g.sum().grad_wrt(&[&seed],false).unwrap()[0].to_vec_f64(),vec![1.;6]);
 }
}
struct Model(Param);
impl ferro_core::nn::Module for Model {
 fn forward(&self,x:&Tensor)->ferro_core::Result<Tensor>{x.mul(&self.0.tensor())}
 fn named_parameters(&self)->Vec<(String,Param)>{vec![("weight".into(),self.0.clone())]}
}
#[test]
fn double_training_checkpoint_preserves_flags_rng_ids_and_subsequent_updates() {
 use ferro_core::{nn::Module,checkpoint::Checkpoint,rng::Rng};
 for adamw in [false,true] {let model=Model(parameter(&[0.7,-0.2]));let rng=Rng::new(7);let mut a=Adam::new(model.parameters(),0.).with_f64_options(0.1234567891234567,0.9,0.999,1e-100,0.01,true,true,None).unwrap();let mut w=AdamW::new(model.parameters(),0.).with_f64_options(0.1234567891234567,0.9,0.999,1e-100,0.01,true,true,None).unwrap();
  for _ in 0..4 {grad(&model.0,&[rng.normal() as f64,rng.normal() as f64]);if adamw{w.step()}else{a.step()}}
  model.0.set_trainable(false);model.0.tensor().set_requires_grad(false).unwrap();let opt:&dyn OptimizerState=if adamw{&w}else{&a};let saved=Checkpoint::from_training_state(4,&model,&[("adam",opt)],&rng).unwrap();
  let path=std::env::temp_dir().join(format!("ferro_adam_f64_{}_{adamw}",std::process::id()));let _=std::fs::remove_dir_all(&path);saved.save_to_dir(&path).unwrap();let loaded=Checkpoint::load_from_dir(&path).unwrap();
  let fresh=Model(parameter(&[8.,9.]));let held=fresh.0.tensor();let id=held.id();let other_rng=Rng::new(99);let mut other_a=Adam::new(fresh.parameters(),0.8);let mut other_w=AdamW::new(fresh.parameters(),0.8);
  let other:&mut dyn OptimizerState=if adamw{&mut other_w}else{&mut other_a};loaded.load_training_state_into(&fresh,&mut[("adam",other)],&other_rng).unwrap();assert_eq!(held.id(),id);assert!(!held.requires_grad());assert!(!fresh.0.is_trainable());assert_eq!(bytes(&saved.tensors),bytes(&Checkpoint::from_training_state(4,&fresh,&[("adam",other)],&other_rng).unwrap().tensors));
  model.0.set_trainable(true);fresh.0.set_trainable(true);model.0.tensor().set_requires_grad(true).unwrap();held.set_requires_grad(true).unwrap();
  for _ in 0..5 {let g=[rng.normal() as f64,rng.normal() as f64];assert_eq!(g,[other_rng.normal() as f64,other_rng.normal() as f64]);grad(&model.0,&g);grad(&fresh.0,&g);if adamw{w.step();other_w.step();assert_eq!(bytes(&w.snapshot()),bytes(&other_w.snapshot()));}else{a.step();other_a.step();assert_eq!(bytes(&a.snapshot()),bytes(&other_a.snapshot()));}assert_eq!(model.0.tensor().to_vec_f64(),held.to_vec_f64());}
  std::fs::remove_dir_all(path).unwrap();
 }
}
#[test]
fn exhausted_later_counter_and_parameter_alias_gradient_reject_before_any_update() {
 for adamw in [false,true] {let a=parameter(&[1.]);let b=parameter(&[2.]);let mut aa=Adam::new(vec![a.clone(),b.clone()],0.1);let mut aw=AdamW::new(vec![a.clone(),b.clone()],0.1);
  let opt:&mut dyn OptimizerState=if adamw{&mut aw}else{&mut aa};let config=opt.configuration();let mut state=opt.snapshot();
  if adamw{state.iter_mut().find(|(k,_)|k=="t").unwrap().1=tensor(&[0.,u32::MAX as f64]);}else{state.iter_mut().find(|(k,_)|k=="t.1").unwrap().1=Tensor::from_vec_f64(vec![u32::MAX as f64],&[]).unwrap();}
  opt.prepare_cpu_restore(&config,&state).unwrap()();let before=bytes(&opt.snapshot());grad(&a,&[0.1]);grad(&b,&[0.2]);let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||if adamw{aw.step()}else{aa.step()}));assert!(result.is_err());assert_eq!(a.tensor().to_vec_f64(),vec![1.]);assert_eq!(b.tensor().to_vec_f64(),vec![2.]);assert_eq!(bytes(&if adamw{aw.snapshot()}else{aa.snapshot()}),before);assert!(a.grad().is_some());assert!(b.grad().is_some());
  // Restore may reset an exhausted counter even with a pending gradient.
  let opt:&mut dyn OptimizerState=if adamw{&mut aw}else{&mut aa};let mut reset=opt.snapshot();for (key,t) in &mut reset{if key=="t"||key.starts_with("t."){*t=Tensor::from_vec_f64(vec![0.;t.numel()],t.shape()).unwrap();}}opt.prepare_cpu_restore(&config,&reset).unwrap()();
  a.tensor().set_leaf_grad(Some(b.tensor().detach())).unwrap();let before=bytes(&opt.snapshot());let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||if adamw{aw.step()}else{aa.step()}));assert!(result.is_err());assert_eq!(bytes(&if adamw{aw.snapshot()}else{aa.snapshot()}),before);assert_eq!(a.tensor().to_vec_f64(),vec![1.]);
 }
}
