use ferro_core::{optim::{Muon,OptimizerState},checkpoint::Checkpoint,nn::Module,Param,Result,Rng,Tensor};
fn config()->Vec<f32>{vec![15.,0.003,0.02,0.95,1.,3.4445,-4.775,2.0315,1e-7,5.,0.]}
fn parameters()->Vec<Param>{vec![Param::new(Tensor::from_vec(vec![0.7,-1.2,0.4,0.2,0.8,-0.6],&[2,3]).unwrap()),Param::new(Tensor::from_vec(vec![0.2,-0.9,0.6,0.4,0.3,-0.5],&[3,2]).unwrap())]}
fn gradient(p:&Param,step:usize){let values=(0..p.tensor().numel()).map(|j|((j+step)%7)as f32*0.125-0.375).collect();p.tensor().backward_with(&Tensor::from_vec(values,p.tensor().shape()).unwrap());}
fn state(o:&Muon)->Vec<(String,Vec<usize>,Vec<u32>)>{o.snapshot().into_iter().map(|(n,t)|(n,t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect()}
fn close(a:&[f32],b:&[f32],tolerance:f32){assert_eq!(a.len(),b.len());for(a,b)in a.iter().zip(b){assert!((a-b).abs()<=tolerance+tolerance*b.abs(),"{a} != {b}");}}
struct Model(Vec<Param>);
impl Module for Model {fn forward(&self,x:&Tensor)->Result<Tensor>{self.0[0].tensor().mul(x)}fn named_parameters(&self)->Vec<(String,Param)>{self.0.iter().enumerate().map(|(i,p)|(format!("p{i}"),p.clone())).collect()}}

#[test]
fn actual_torch_twenty_updates_momentum_skips_and_owning_snapshots(){
 let p=parameters();let pointers:Vec<_>=p.iter().map(|p|p.tensor()._storage_ptr()).collect();
 let mut c=config();let mut o=Muon::new(p.clone(),&c).unwrap();
 for step in 0..20{if step==10{c[1]=0.0015;o.restore_configuration(&c).unwrap();}gradient(&p[0],step);if step%5!=2{gradient(&p[1],step);}o.step().unwrap();}
 close(&p[0].tensor().to_vec(),&[0.71842426,-1.19682443,0.39326945,0.21039340,0.79760844,-0.60619020],3e-5);
 close(&p[1].tensor().to_vec(),&[0.20981562,-0.90318435,0.59901631,0.39646611,0.30959332,-0.50934136],3e-5);
 let saved=o.snapshot();let get=|name:&str|saved.iter().find(|(n,_)|n==name).unwrap().1.to_vec();
 close(&get("momentum_buffer.0"),&[-0.001973246,0.023597028,0.008148251,-0.0025186224,-0.008642691,-0.010451096],1e-6);
 close(&get("momentum_buffer.1"),&[0.0008168379,0.010753347,0.012102511,-0.015276848,0.0016963471,0.0031714963],1e-6);
 assert_eq!(get("initialized.0"),vec![1.]);assert_eq!(get("initialized.1"),vec![1.]);
 assert_eq!(p.iter().map(|p|p.tensor()._storage_ptr()).collect::<Vec<_>>(),pointers);
 let clone=o.clone();let before=state(&o);gradient(&p[0],21);o.step().unwrap();assert_eq!(state(&clone),before);
 assert_eq!(saved.iter().map(|(n,t)|(n.clone(),t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect::<Vec<_>>(),before);
}

#[test]
fn exact_native_checkpoint_and_atomic_late_state_corruption(){
 let source=Model(parameters());let mut o=Muon::new(source.parameters(),&config()).unwrap();let rng=Rng::new(17);
 for step in 0..5{for p in &source.0{gradient(p,step);}o.step().unwrap();}
 let saved=Checkpoint::from_training_state(5,&source,&[("main",&o)],&rng).unwrap();
 let target=Model(parameters());let mut other=Muon::new(target.parameters(),&config()).unwrap();let other_rng=Rng::new(99);
 saved.load_training_state_into(&target,&mut[("main",&mut other)],&other_rng).unwrap();
 for step in 5..12{for p in source.0.iter().chain(&target.0){gradient(p,step);}o.step().unwrap();other.step().unwrap();}
 assert_eq!(state(&o),state(&other));assert_eq!(source.0.iter().map(|p|p.tensor().to_vec()).collect::<Vec<_>>(),target.0.iter().map(|p|p.tensor().to_vec()).collect::<Vec<_>>());
 for field in ["optim.main.momentum_buffer.1","optim.main.initialized.1","optim.main.unexpected"] {
  let mut malformed=saved.clone();if let Some((_,t))=malformed.tensors.iter_mut().find(|(n,_)|n==field){*t=Tensor::zeros(&[99]);}else{malformed.tensors.push((field.into(),Tensor::scalar(1.)));}
  let before=(state(&other),other.configuration(),other_rng.state(),target.0.iter().map(|p|(p.tensor().to_vec(),p.tensor()._version())).collect::<Vec<_>>());
  assert!(malformed.load_training_state_into(&target,&mut[("main",&mut other)],&other_rng).is_err());
  assert_eq!((state(&other),other.configuration(),other_rng.state(),target.0.iter().map(|p|(p.tensor().to_vec(),p.tensor()._version())).collect::<Vec<_>>()),before);
 }
 let before=state(&other);drop(other.prepare_cpu_restore(&config(),&o.snapshot()).unwrap());assert_eq!(state(&other),before);
 let mut bad=other.snapshot();bad.iter_mut().find(|(n,_)|n=="initialized.1").unwrap().1=Tensor::scalar(0.);
 assert!(other.restore(&bad).is_err());assert_eq!(state(&other),before);
}

#[test]
fn shape_and_capture_preflight_empty_rows_and_frozen_skips(){
 for shape in [vec![3],vec![2,0],vec![0,0],vec![1usize<<62,0]]{
  let n=shape.iter().product();assert!(Muon::new(vec![Param::new(Tensor::from_vec(vec![0.;n],&shape).unwrap())],&config()).is_err());
 }
 let p=parameters();let empty=Param::new(Tensor::zeros(&[0,3]));let mut params=p.clone();params.push(empty.clone());
 let mut o=Muon::new(params.clone(),&config()).unwrap();let before=state(&o);
 p[0].tensor().set_leaf_grad(Some(p[0].tensor().detach())).unwrap();
 assert!(o.step().is_err());assert_eq!(state(&o),before);o.zero_grad();
 assert!(ferro_core::capture(||o.step()).is_err());assert_eq!(state(&o),before);
 gradient(&p[0],0);empty.tensor().backward_with(&Tensor::zeros(&[0,3]));o.step().unwrap();
 assert_eq!(o.snapshot().iter().find(|(n,_)|n=="initialized.2").unwrap().1.item(),1.);
 p[0].set_trainable(false);gradient(&p[0],1);let before=state(&o);let values=p[0].tensor().to_vec();o.step().unwrap();
 assert_eq!(state(&o),before);assert_eq!(p[0].tensor().to_vec(),values);assert!(p[0].tensor().grad().is_some());
 let mut bad=config();bad[9]=100.;assert!(o.restore_configuration(&bad).is_err());assert_eq!(state(&o),before);
 let mut bad=config();bad[8]=3.39e38;assert!(o.restore_configuration(&bad).is_err());assert_eq!(state(&o),before);
}

#[test]
fn shared_parameter_leaves_match_separate_ordered_optimizers(){
 let table=Tensor::from_vec(vec![0.7,-1.2,0.4,0.2,0.8,-0.6],&[2,3]).unwrap();
 let control=Tensor::from_vec(table.to_vec(),table.shape()).unwrap();
 let params:Vec<_>=(0..2).map(|_|Param::from_shared(&table,true).unwrap()).collect();
 let separate:Vec<_>=(0..2).map(|_|Param::from_shared(&control,true).unwrap()).collect();
 let mut together=Muon::new(params.clone(),&config()).unwrap();
 let mut independent:Vec<_>=separate.iter().map(|p|Muon::new(vec![p.clone()],&config()).unwrap()).collect();
 for step in 0..12 {
  for i in 0..2 {if (step+i)%5!=1{gradient(&params[i],step+i);gradient(&separate[i],step+i);}}
  together.step().unwrap();for o in &mut independent{o.step().unwrap();}
  assert_eq!(table.to_vec(),control.to_vec());
  let snapshot=together.snapshot();
  for (i,o) in independent.iter().enumerate(){for(name,t)in o.snapshot(){
   let key=name.strip_suffix(".0").unwrap().to_owned()+&format!(".{i}");
   assert_eq!(snapshot.iter().find(|(n,_)|n==&key).unwrap().1.to_vec(),t.to_vec());
  }}
 }
}
