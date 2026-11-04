use ferro_core::{checkpoint::Checkpoint, nn::Module, optim::{Lbfgs,OptimizerState}, Param, Result, Rng, Tensor};
fn config()->Vec<f32> {vec![13.,0.4,3.,4.,0.,0.,3.]}
fn parameter()->Param {Param::new(Tensor::from_vec(vec![1.5,-0.7,0.2],&[3]).unwrap())}
fn loss(p:&Param,shift:f32)->Result<f32> {
    p.zero_grad();let x=p.tensor();
    let value=x.square()?.mul(&Tensor::from_vec(vec![0.7,1.1,1.4],&[3])?)?.sum()
        .add(&x.powf(4.).sum().mul(&Tensor::scalar(0.1))?)?.add(&x.sum().mul(&Tensor::scalar(shift))?)?;
    let number=value.item();value.backward();Ok(number)
}
fn state(o:&Lbfgs)->Vec<(String,Vec<usize>,Vec<u32>)> {o.snapshot().into_iter().map(|(n,t)|(n,t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect()}
fn close(a:&[f32],b:&[f32]) {assert_eq!(a.len(),b.len());for(a,b)in a.iter().zip(b){assert!((a-b).abs()<=3e-6+3e-5*b.abs(),"{a} != {b}");}}
struct Model(Param);
impl Module for Model {
 fn forward(&self,x:&Tensor)->Result<Tensor>{self.0.tensor().mul(x)}
 fn named_parameters(&self)->Vec<(String,Param)>{vec![("weight".into(),self.0.clone())]}
}
#[test]
fn fixed_step_history_matches_actual_torch_and_owns_snapshots() {
 let p=parameter();let pointer=p.tensor()._storage_ptr();let mut o=Lbfgs::new(vec![p.clone()],&config()).unwrap();let mut losses=vec![];let mut calls=0;
 for step in 0..7 {losses.push(o.step(||{calls+=1;loss(&p,-0.2+0.025*step as f32)}).unwrap());}
 assert_eq!(calls,21);assert_eq!(pointer,p.tensor()._storage_ptr());assert!(p.grad().is_none());
 close(&p.tensor().to_vec(),&[0.040554676,0.025936738,0.020172203]);
 close(&losses,&[2.5004199,0.40040666,0.021181844,-0.009207975,-0.0073115155,-0.0035243668,-0.0011488595]);
 let arrays=o.snapshot();let get=|n:&str|arrays.iter().find(|(k,_)|k==n).unwrap().1.to_vec();
 assert_eq!(get("header"),vec![21.,21.,1.,1.,3.]);
 close(&get("d"),&[-0.008052556,-0.0053038173,-0.003945695]);
 close(&get("prev_grad"),&[0.011319533,0.011737019,0.010905463]);
 close(&get("scalars"),&[0.4,0.5132643,-0.0018090429]);
 let cloned=o.clone();let before=state(&o);o.step(||loss(&p,0.1)).unwrap();assert_eq!(state(&cloned),before);
 assert_eq!(arrays.iter().map(|(n,t)|(n.clone(),t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect::<Vec<_>>(),before);
}
#[test]
fn checkpoint_restores_every_history_field_and_rejects_late_corruption_atomically() {
 let source=Model(parameter());let mut o=Lbfgs::new(source.parameters(),&config()).unwrap();let rng=Rng::new(13);
 for i in 0..4{o.step(||loss(&source.0,-0.2+0.025*i as f32)).unwrap();}
 let saved=Checkpoint::from_training_state(4,&source,&[("main",&o)],&rng).unwrap();
 let target=Model(parameter());let mut other=Lbfgs::new(target.parameters(),&config()).unwrap();let other_rng=Rng::new(99);
 saved.load_training_state_into(&target,&mut[("main",&mut other)],&other_rng).unwrap();
 for i in 4..9 {assert_eq!(o.step(||loss(&source.0,-0.2+0.025*i as f32)).unwrap(),other.step(||loss(&target.0,-0.2+0.025*i as f32)).unwrap());}
 assert_eq!(source.0.tensor().to_vec(),target.0.tensor().to_vec());assert_eq!(state(&o),state(&other));assert_eq!(rng.state(),other_rng.state());
 for fault in 0..5 {
  let mut bad=saved.clone();match fault {
   0=>bad.tensors.push(("optim.main.unexpected".into(),Tensor::scalar(1.))),
   1=>bad.tensors.iter_mut().find(|(n,_)|n=="optim.main.alpha_valid").unwrap().1=Tensor::from_vec(vec![1.,0.5,1.],&[3]).unwrap(),
   2=>bad.tensors.iter_mut().find(|(n,_)|n=="optim.main.rho").unwrap().1=Tensor::from_vec(vec![1.,f32::NAN,1.],&[3]).unwrap(),
   3=>bad.tensors.iter_mut().find(|(n,_)|n=="optim.main.config").unwrap().1=Tensor::from_vec(vec![13.,0.4,3.,4.,0.,0.,2.],&[7]).unwrap(),
   _=>bad.tensors.iter_mut().find(|(n,_)|n=="optim.main.header").unwrap().1=Tensor::from_vec(vec![12.,12.,0.,1.,3.],&[5]).unwrap(),
  }
  let before=(target.0.tensor().to_vec(),target.0.tensor()._version(),state(&other),other.configuration(),other_rng.state());
  assert!(bad.load_training_state_into(&target,&mut[("main",&mut other)],&other_rng).is_err());
  assert_eq!((target.0.tensor().to_vec(),target.0.tensor()._version(),state(&other),other.configuration(),other_rng.state()),before);
 }
 let before=state(&other);let prepared=other.prepare_cpu_restore(&config(),&o.snapshot()).unwrap();drop(prepared);assert_eq!(state(&other),before);
}
#[test]
fn zero_gradient_and_boundary_preflight_preserve_parameters() {
 let p=parameter();let mut o=Lbfgs::new(vec![p.clone()],&config()).unwrap();let original=p.tensor().to_vec();
 assert_eq!(o.step(||{p.zero_grad();Ok(2.)}).unwrap(),2.);assert_eq!(p.tensor().to_vec(),original);
 let saved=o.snapshot();o.restore(&saved).unwrap();assert_eq!(o.snapshot()[0].1.to_vec(),vec![1.,0.,0.,0.,0.]);
 let before=state(&o);p.set_trainable(false);assert!(o.step(||panic!("must reject before closure")).is_err());p.set_trainable(true);assert_eq!(state(&o),before);
 assert!(ferro_core::capture(||o.step(||panic!("capture preflight"))).is_err());assert_eq!(state(&o),before);
 assert!(Lbfgs::new(vec![],&config()).is_err());
 let mut bad=config();bad[6]=0.;assert!(Lbfgs::new(vec![p.clone()],&bad).is_err());
 assert!(o.step(||Ok(f32::NAN)).is_err());assert_eq!(state(&o),before);
 let mut exhausted=saved.clone();exhausted.iter_mut().find(|(name,_)|name=="header").unwrap().1=Tensor::from_vec(vec![(1u32<<24) as f32,0.,0.,0.,0.],&[5]).unwrap();
 o.restore(&exhausted).unwrap();let before=state(&o);
 assert!(o.step(||panic!("counter capacity must precede closure")).is_err());assert_eq!(state(&o),before);
}

fn rosenbrock(p:&Param)->Result<f32> {
 p.zero_grad();let x=p.tensor().to_vec();let error=x[1]-x[0]*x[0];
 p.tensor().backward_with(&Tensor::from_vec(vec![-400.*x[0]*error-2.*(1.-x[0]),200.*error],&[2])?);
 Ok(100.*error*error+(1.-x[0]).powi(2))
}
#[test]
fn strong_wolfe_rosenbrock_matches_torch_and_first_step_satisfies_both_conditions() {
 let p=Param::new(Tensor::from_vec(vec![-1.2,1.],&[2]).unwrap());
 let mut o=Lbfgs::new(vec![p.clone()],&[13.,0.8,4.,8.,0.,0.,4.,1.]).unwrap();
 let expected=[([-1.02780294,1.05878127],6.),([-0.9079548,0.7817270],11.),([-0.43208194,0.17209297],17.)];
 for (parameters,calls) in expected {
  o.step(||rosenbrock(&p)).unwrap();
  for (actual,target) in p.tensor().to_vec().iter().zip(parameters){assert!((actual-target).abs()<3e-4,"{actual} != {target}");}
  assert_eq!(o.snapshot()[0].1.to_vec()[0],calls);
 }
 let p=Param::new(Tensor::from_vec(vec![-1.2,1.],&[2]).unwrap());let mut o=Lbfgs::new(vec![p.clone()],&[13.,1.,1.,25.,0.,0.,4.,1.]).unwrap();
 let initial=rosenbrock(&p).unwrap();let gradient=p.grad().unwrap().to_vec();o.step(||rosenbrock(&p)).unwrap();
 let next=rosenbrock(&p).unwrap();let next_gradient=p.grad().unwrap().to_vec();let t=o.snapshot().iter().find(|(n,_)|n=="scalars").unwrap().1.to_vec()[0];
 let slope=-gradient.iter().map(|g|g*g).sum::<f32>();let next_slope=-gradient.iter().zip(next_gradient).map(|(a,b)|a*b).sum::<f32>();
 assert!(next<=initial+1e-4*t*slope);assert!(next_slope.abs()<=-0.9*slope);
 assert!((t-0.0008138593).abs()<1e-8);
}
#[test]
fn search_mode_restores_into_either_constructed_mode_and_rejects_invalid_flags() {
 for search in [false,true] {
  let source=Model(parameter());let mut c=config();if search{c.push(1.);}let mut o=Lbfgs::new(source.parameters(),&c).unwrap();let rng=Rng::new(7);
  for i in 0..3{o.step(||loss(&source.0,-0.2+0.025*i as f32)).unwrap();}
  let saved=Checkpoint::from_training_state(3,&source,&[("main",&o)],&rng).unwrap();
  let target=Model(parameter());let mut other_c=config();if !search{other_c.push(1.);}let mut other=Lbfgs::new(target.parameters(),&other_c).unwrap();let other_rng=Rng::new(99);
  saved.load_training_state_into(&target,&mut[("main",&mut other)],&other_rng).unwrap();assert_eq!(other.configuration(),c);
  for i in 3..7{assert_eq!(o.step(||loss(&source.0,-0.2+0.025*i as f32)).unwrap(),other.step(||loss(&target.0,-0.2+0.025*i as f32)).unwrap());}
  assert_eq!(state(&o),state(&other));assert_eq!(source.0.tensor().to_vec(),target.0.tensor().to_vec());
  let mut malformed=saved.clone();let mut invalid=config();invalid.push(2.);
  malformed.tensors.iter_mut().find(|(n,_)|n=="optim.main.config").unwrap().1=Tensor::from_vec(invalid,&[8]).unwrap();
  let before=(state(&other),target.0.tensor().to_vec(),other_rng.state());assert!(malformed.load_training_state_into(&target,&mut[("main",&mut other)],&other_rng).is_err());
  assert_eq!((state(&other),target.0.tensor().to_vec(),other_rng.state()),before);
 }
}
