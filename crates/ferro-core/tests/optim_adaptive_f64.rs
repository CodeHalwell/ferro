use ferro_core::{Tensor,Param,DType,optim::{Adaptive,OptimizerState}};
include!("fixtures/adaptive_f64_goldens.rs");
fn tensor(v:&[f64])->Tensor{Tensor::from_vec_f64(v.to_vec(),&[v.len()]).unwrap()}
fn parameter(v:&[f64])->Param{Param::new(tensor(v))}
fn gradient(p:&Param,g:&[f64]){p.tensor().set_leaf_grad(Some(tensor(g))).unwrap();}
fn exact(t:&Tensor,expected:&[u64],message:&str){assert_eq!(t.numel(),expected.len());for (i,(x,&bits))in t.to_vec_f64().iter().zip(expected).enumerate(){let y=f64::from_bits(bits);assert!(x.to_bits()==bits||x.is_nan()&&y.is_nan(),"{message} at{i}: {x:?}/{:x} != {y:?}/{bits:x}",x.to_bits());}}
fn bytes(s:&[(String,Tensor)])->Vec<u8>{ferro_core::safetensors::to_safetensors_bytes(&s.iter().map(|(k,v)|(k.as_str(),v)).collect::<Vec<_>>()).unwrap()}
#[test]
fn actual_torch_132_configurations_match_double_parameters_and_every_state_bit(){
 for (ci,c)in GOLDENS.iter().enumerate(){let initial:Vec<_>=c.initial.iter().map(|&x|f64::from_bits(x)).collect();let p=parameter(&initial);let held=p.tensor();let id=held.id();let storage=held._storage_ptr();let mut opt=Adaptive::new_f64(vec![p.clone()],c.kind,c.values.map(f64::from_bits),c.centered,c.maximize).unwrap();
  if c.kind==6{exact(&opt.snapshot().iter().find(|(k,_)|k=="sum.0").unwrap().1,&vec![c.values[3];initial.len()],"initial accumulator");}
  for (si,(g,expected))in c.gradients.iter().zip(c.steps).enumerate(){gradient(&p,&g.iter().map(|&x|f64::from_bits(x)).collect::<Vec<_>>());opt.step().unwrap();exact(&held,expected.parameter,&format!("case{ci} step{si} parameter"));
   let names:&[&str]=if c.kind==5{&["square_avg.0","momentum_buffer.0","grad_avg.0"]}else{&["sum.0"]};let state=opt.snapshot();
   for (name,values)in names.iter().zip(expected.state){let value=&state.iter().find(|(k,_)|k==name).unwrap().1;assert_eq!(value.dtype(),DType::F64);exact(value,values,&format!("case{ci} step{si} {name}"));}
   assert_eq!(p.tensor().id(),id);assert_eq!(p.tensor()._storage_ptr(),storage);assert!(p.grad().is_none());
  }
 }
}

fn configured(params:Vec<Param>, kind:u8)->Adaptive {
    let values=if kind==5 {[0.1234567891234567,0.9123456789123456,1e-100,0.0234567891234567,0.7123456789123456]}
               else {[0.1234567891234567,0.1234567891234567,0.0234567891234567,1e-200,1e-100]};
    Adaptive::new_f64(params,kind,values,kind==5,true).unwrap()
}
#[test]
fn exact_configuration_and_late_typed_restore_errors_preserve_every_slot() {
    for kind in [5,6] {
        let a=parameter(&[0.7,-0.2]);let b=parameter(&[0.3,0.4]);let mut opt=configured(vec![a.clone(),b.clone()],kind);
        gradient(&a,&[0.2,0.1]);gradient(&b,&[-0.3,0.5]);opt.step().unwrap();
        let config=opt.configuration();assert_eq!(config.len(),25);let state=opt.snapshot();let before=bytes(&state);
        for chunk in [f32::NAN,-1.,65536.,0.5] {let mut bad=config.clone();bad[17]=chunk;
            assert!(opt.restore_configuration(&bad).is_err());assert_eq!(opt.configuration(),config);assert_eq!(bytes(&opt.snapshot()),before);}
        for value in [Tensor::zeros(&[2]),tensor(&[0.])] {let mut bad=state.clone();bad.last_mut().unwrap().1=value;
            assert!(opt.prepare_cpu_restore(&config,&bad).is_err());assert_eq!(bytes(&opt.snapshot()),before);}
        opt.prepare_cpu_restore(&config,&state).unwrap()();assert_eq!(bytes(&opt.snapshot()),before);
        let state_handles=opt.state_buffers().unwrap();let ptrs:Vec<_>=state_handles.iter().map(Tensor::_storage_ptr).collect();
        opt.set_lr_f64(1e-200).unwrap();assert_eq!(opt.lr_f64().to_bits(),1e-200f64.to_bits());
        assert_eq!(opt.state_buffers().unwrap().iter().map(Tensor::_storage_ptr).collect::<Vec<_>>(),ptrs);
    }
}
#[test]
fn mixed_members_keep_legacy_float32_updates_and_validate_before_commit() {
    for kind in [5,6] {
        let a=Param::new(Tensor::from_vec(vec![0.7,-0.2],&[2]).unwrap());let b=parameter(&[0.3,0.4]);let control=Param::new(a.tensor());
        let legacy=if kind==5{vec![5.,0.017,0.7,1e-8,0.03,0.9,1.,0.]}else{vec![6.,0.017,0.03,0.03,0.7,1e-8,0.]};
        let values=[legacy[1] as f64,legacy[2] as f64,legacy[3] as f64,legacy[4] as f64,legacy[5] as f64];
        let mut mixed=Adaptive::new_f64(vec![a.clone(),b.clone()],kind,values,kind==5,false).unwrap();
        let mut single=Adaptive::new(vec![control.clone()],&legacy).unwrap();
        for step in 0..6 {let g=Tensor::from_vec(vec![0.2+step as f32*0.03,-0.4],&[2]).unwrap();a.tensor().set_leaf_grad(Some(g.clone())).unwrap();control.tensor().set_leaf_grad(Some(g)).unwrap();gradient(&b,&[0.7,-0.2]);mixed.step().unwrap();single.step().unwrap();
            assert_eq!(a.tensor().to_vec(),control.tensor().to_vec());let state=mixed.snapshot();
            for (key,value)in single.snapshot(){assert_eq!(value.to_vec(),state.iter().find(|(k,_)|k==&key).unwrap().1.to_vec());}
        }
        let old=a.tensor().to_vec();let buffers=||mixed.state_buffers().unwrap().into_iter().enumerate().map(|(i,t)|(i.to_string(),t)).collect::<Vec<_>>();let before=bytes(&buffers());b.set(tensor(&[0.3]));a.tensor().set_leaf_grad(Some(Tensor::ones(&[2]))).unwrap();gradient(&b,&[0.1]);
        assert!(mixed.step().is_err());assert_eq!(old,a.tensor().to_vec());assert_eq!(before,bytes(&mixed.state_buffers().unwrap().into_iter().enumerate().map(|(i,t)|(i.to_string(),t)).collect::<Vec<_>>()));assert!(a.grad().is_some());
        assert!(Adaptive::new_f64(vec![a],kind,[1e200,0.3,0.,0.,0.],false,false).is_err());
    }
}
#[test]
fn shared_parameter_cells_advance_in_order_with_independent_state() {
    for kind in [5,6] {let data=tensor(&[0.7,-0.4]);let other=tensor(&[0.7,-0.4]);
        let params:Vec<_>=(0..2).map(|_|Param::from_shared(&data,true).unwrap()).collect();let controls:Vec<_>=(0..2).map(|_|Param::from_shared(&other,true).unwrap()).collect();
        let mut mixed=configured(params.clone(),kind);let mut singles:Vec<_>=controls.iter().map(|p|configured(vec![p.clone()],kind)).collect();
        for step in 0..7 {for i in 0..2{if (step+i)%3!=1{let g=[step as f64*0.13+i as f64,-0.2];gradient(&params[i],&g);gradient(&controls[i],&g);}}
            mixed.step().unwrap();for o in &mut singles{o.step().unwrap();}assert_eq!(data.to_vec_f64(),other.to_vec_f64());
            let state=mixed.snapshot();for (i,single)in singles.iter().enumerate(){for(key,value)in single.snapshot(){let name=format!("{}.{i}",key.rsplit_once('.').unwrap().0);let actual=&state.iter().find(|(k,_)|k==&name).unwrap().1;if key.starts_with("t."){assert_eq!(actual.item(),value.item());}else{assert_eq!(actual.to_vec_f64(),value.to_vec_f64());}}}
        }
    }
}
#[test]
fn counter_limit_alias_and_mode_errors_precede_updates_and_allow_restore() {
    use ferro_core::capture::capture;
    for kind in [5,6] {let a=parameter(&[0.7]);let b=parameter(&[0.3]);let mut opt=configured(vec![a.clone(),b.clone()],kind);let config=opt.configuration();
        let mut state=opt.snapshot();state.iter_mut().find(|(k,_)|k=="t.1").unwrap().1=Tensor::scalar((1u32<<24)as f32);opt.prepare_cpu_restore(&config,&state).unwrap()();
        gradient(&a,&[0.1]);gradient(&b,&[0.2]);let before=bytes(&opt.snapshot());assert!(opt.step().is_err());assert_eq!(a.tensor().to_vec_f64(),vec![0.7]);assert_eq!(bytes(&opt.snapshot()),before);assert!(a.grad().is_some());
        for (k,v)in &mut state{if k.starts_with("t."){*v=Tensor::scalar(0.);}}opt.prepare_cpu_restore(&config,&state).unwrap()();
        let before=bytes(&opt.snapshot());assert!(capture(||opt.step()).is_err());assert_eq!(bytes(&opt.snapshot()),before);
        a.tensor().set_leaf_grad(Some(b.tensor().detach())).unwrap();assert!(opt.step().is_err());assert_eq!(bytes(&opt.snapshot()),before);
        gradient(&a,&[0.1]);opt.step().unwrap();assert!(a.grad().is_none());
    }
}
#[test]
fn eager_accumulator_empty_scalar_and_live_strided_gradient_contracts() {
    for kind in [5,6] {
        for shape in [vec![],vec![0,3]]{let p=Param::new(Tensor::from_vec_f64(vec![0.7;shape.iter().product()],&shape).unwrap());let mut opt=configured(vec![p.clone()],kind);let before=bytes(&opt.snapshot());opt.step().unwrap();assert_eq!(bytes(&opt.snapshot()),before);
            p.tensor().set_leaf_grad(Some(Tensor::from_vec_f64(vec![0.2;p.tensor().numel()],&shape).unwrap())).unwrap();opt.step().unwrap();assert_eq!(opt.snapshot()[0].1.item(),1.);}
        let p=Param::new(Tensor::from_vec_f64(vec![1.;6],&[3,2]).unwrap());let seed=Tensor::from_vec_f64(vec![0.1,0.2,0.3,0.4,0.5,0.6],&[2,3]).unwrap();seed.set_requires_grad(true).unwrap();let g=seed.transpose(0,1).unwrap();let id=g.id();let storage=g._storage_ptr();p.tensor().set_leaf_grad(Some(g.clone())).unwrap();configured(vec![p.clone()],kind).step().unwrap();
        assert_eq!(g.id(),id);assert_eq!(g._storage_ptr(),storage);assert_eq!(g.strides(),&[1,3]);assert!(g.requires_grad());assert_eq!(g.sum().grad_wrt(&[&seed],false).unwrap()[0].to_vec_f64(),vec![1.;6]);assert!(p.grad().is_none());
    }
    let p=parameter(&[0.]);let mut opt=Adaptive::new_f64(vec![p],6,[0.1,0.,0.,1e-300,1e-100],false,false).unwrap();let before=bytes(&opt.snapshot());opt.restore_configuration(&Adaptive::new_f64(vec![],6,[0.1,0.,0.,0.9,1e-100],false,false).unwrap().configuration()).unwrap();assert_eq!(bytes(&opt.snapshot()),before);
}
struct Model(Param);
impl ferro_core::nn::Module for Model {
    fn forward(&self,x:&Tensor)->ferro_core::Result<Tensor>{x.mul(&self.0.tensor())}
    fn named_parameters(&self)->Vec<(String,Param)>{vec![("weight".into(),self.0.clone())]}
}
#[test]
fn complete_typed_checkpoint_restores_flags_rng_and_exact_continuation() {
    use ferro_core::{checkpoint::Checkpoint,nn::Module,rng::Rng};
    for kind in [5,6]{let model=Model(parameter(&[0.7,-0.3]));let rng=Rng::new(7);let mut opt=configured(model.parameters(),kind);
        for _ in 0..3{gradient(&model.0,&[rng.normal()as f64,rng.normal()as f64]);opt.step().unwrap();}model.0.set_trainable(false);model.0.tensor().set_requires_grad(false).unwrap();
        let saved=Checkpoint::from_training_state(3,&model,&[("adaptive",&opt)],&rng).unwrap();let path=std::env::temp_dir().join(format!("ferro_adaptive_f64_{}_{kind}",std::process::id()));let _=std::fs::remove_dir_all(&path);saved.save_to_dir(&path).unwrap();let loaded=Checkpoint::load_from_dir(&path).unwrap();
        let fresh=Model(parameter(&[8.,9.]));let held=fresh.0.tensor();let id=held.id();let other_rng=Rng::new(99);let mut other=configured(fresh.parameters(),kind);loaded.load_training_state_into(&fresh,&mut[("adaptive",&mut other)],&other_rng).unwrap();assert_eq!(held.id(),id);assert!(!held.requires_grad());assert!(!fresh.0.is_trainable());assert_eq!(bytes(&saved.tensors),bytes(&Checkpoint::from_training_state(3,&fresh,&[("adaptive",&other)],&other_rng).unwrap().tensors));
        model.0.set_trainable(true);fresh.0.set_trainable(true);model.0.tensor().set_requires_grad(true).unwrap();held.set_requires_grad(true).unwrap();
        for _ in 0..5{let g=[rng.normal()as f64,rng.normal()as f64];assert_eq!(g,[other_rng.normal()as f64,other_rng.normal()as f64]);gradient(&model.0,&g);gradient(&fresh.0,&g);opt.step().unwrap();other.step().unwrap();assert_eq!(model.0.tensor().to_vec_f64(),held.to_vec_f64());assert_eq!(bytes(&opt.snapshot()),bytes(&other.snapshot()));}std::fs::remove_dir_all(path).unwrap();
    }
}
#[test]
fn finite_scalar_policy_and_unsupported_families_remain_explicit() {
    let p=parameter(&[0.7]);for kind in [5,6]{for value in [f64::NAN,f64::INFINITY,-0.1]{assert!(Adaptive::new_f64(vec![p.clone()],kind,[value,0.,0.,0.,0.],false,false).is_err());}}
    assert!(Adaptive::new_f64(vec![p.clone()],5,[0.1,1.1,0.,0.,0.],false,false).is_ok());
    for config in [vec![7.,0.01,0.9,0.99,1e-8,0.,0.],vec![8.,0.1,0.9,1e-8,0.,0.]]{assert!(Adaptive::new(vec![p.clone()],&config).is_err());}
}
