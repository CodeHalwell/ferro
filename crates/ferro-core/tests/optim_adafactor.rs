use ferro_core::{optim::{Adafactor,OptimizerState},checkpoint::Checkpoint,nn::Module,Param,Result,Rng,Tensor};
fn config()->Vec<f32>{vec![14.,0.03,-0.8,-1.,0.001,1.2,0.02,1.]}
fn parameters()->Vec<Param>{vec![Param::new(Tensor::from_vec(vec![0.7,-1.2,0.4,0.2,0.8,-0.6],&[2,3]).unwrap()),Param::new(Tensor::from_vec(vec![0.2,-0.9,0.6],&[3]).unwrap())]}
fn gradient(p:&Param,step:usize){let values=(0..p.tensor().numel()).map(|j|((j+step)%7)as f32*0.125-0.375).collect();p.tensor().backward_with(&Tensor::from_vec(values,p.tensor().shape()).unwrap());}
fn state(o:&Adafactor)->Vec<(String,Vec<usize>,Vec<u32>)>{o.snapshot().into_iter().map(|(n,t)|(n,t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect()}
fn close(a:&[f32],b:&[f32]){assert_eq!(a.len(),b.len());for(a,b)in a.iter().zip(b){assert!((a-b).abs()<=3e-6+3e-5*b.abs(),"{a} != {b}");}}
struct Model(Vec<Param>);
impl Module for Model {fn forward(&self,x:&Tensor)->Result<Tensor>{self.0[0].tensor().mul(x)}fn named_parameters(&self)->Vec<(String,Param)>{self.0.iter().enumerate().map(|(i,p)|(format!("p{i}"),p.clone())).collect()}}
#[test]
fn actual_torch_matrix_and_vector_trajectory_and_all_factors() {
 let p=parameters();let pointers:Vec<_>=p.iter().map(|p|p.tensor()._storage_ptr()).collect();let mut c=config();let mut o=Adafactor::new(p.clone(),&c).unwrap();
 for step in 0..20{if step==10{c[1]=0.015;o.restore_configuration(&c).unwrap();}gradient(&p[0],step);if step%5!=2{gradient(&p[1],step);}o.step().unwrap();}
 close(&p[0].tensor().to_vec(),&[0.64995557,-1.17351747,0.43163437,0.22123592,0.8494507,-0.5511282]);
 close(&p[1].tensor().to_vec(),&[0.19951454,-0.8807100,0.6223021]);
 let saved=o.snapshot();let get=|name:&str|saved.iter().find(|(n,_)|n==name).unwrap().1.to_vec();
 close(&get("row_var.0"),&[0.06210692,0.06516853]);close(&get("col_var.0"),&[0.062375277,0.06380443,0.06473346]);
 close(&get("variance.1"),&[0.0571199,0.062027726,0.068177745]);assert_eq!(get("t.0"),vec![20.]);assert_eq!(get("t.1"),vec![16.]);
 assert_eq!(get("variance.0"),Vec::<f32>::new());assert_eq!(get("row_var.1"),Vec::<f32>::new());
 assert_eq!(o.state_buffers().unwrap().iter().map(Tensor::numel).sum::<usize>(),8);
 assert_eq!(p.iter().map(|p|p.tensor()._storage_ptr()).collect::<Vec<_>>(),pointers);
 let clone=o.clone();let before=state(&o);gradient(&p[0],21);o.step().unwrap();assert_eq!(state(&clone),before);
 assert_eq!(saved.iter().map(|(n,t)|(n.clone(),t.shape().to_vec(),t.to_vec().into_iter().map(f32::to_bits).collect())).collect::<Vec<_>>(),before);
}
#[test]
fn exact_checkpoint_factor_restore_and_late_corruption_transaction() {
 let source=Model(parameters());let mut o=Adafactor::new(source.parameters(),&config()).unwrap();let rng=Rng::new(17);
 for step in 0..5{for p in &source.0{gradient(p,step);}o.step().unwrap();}
 let saved=Checkpoint::from_training_state(5,&source,&[("main",&o)],&rng).unwrap();
 let target=Model(parameters());let mut other=Adafactor::new(target.parameters(),&config()).unwrap();let other_rng=Rng::new(99);
 saved.load_training_state_into(&target,&mut[("main",&mut other)],&other_rng).unwrap();
 for step in 5..12{for p in source.0.iter().chain(&target.0){gradient(p,step);}o.step().unwrap();other.step().unwrap();}
 assert_eq!(state(&o),state(&other));assert_eq!(source.0.iter().map(|p|p.tensor().to_vec()).collect::<Vec<_>>(),target.0.iter().map(|p|p.tensor().to_vec()).collect::<Vec<_>>());
 for field in ["optim.main.col_var.0","optim.main.variance.1","optim.main.t.1","optim.main.unexpected"] {
  let mut malformed=saved.clone();if let Some((_,t))=malformed.tensors.iter_mut().find(|(n,_)|n==field){*t=Tensor::zeros(&[99]);}else{malformed.tensors.push((field.into(),Tensor::scalar(1.)));}
  let before=(state(&other),other.configuration(),other_rng.state(),target.0.iter().map(|p|(p.tensor().to_vec(),p.tensor()._version())).collect::<Vec<_>>());
  assert!(malformed.load_training_state_into(&target,&mut[("main",&mut other)],&other_rng).is_err());
  assert_eq!((state(&other),other.configuration(),other_rng.state(),target.0.iter().map(|p|(p.tensor().to_vec(),p.tensor()._version())).collect::<Vec<_>>()),before);
 }
 let before=state(&other);drop(other.prepare_cpu_restore(&config(),&o.snapshot()).unwrap());assert_eq!(state(&other),before);
}
#[test]
fn empty_active_gradients_capture_and_counter_overflow_reject_before_updates() {
 let p=parameters();let empty=Param::new(Tensor::zeros(&[0,3]));let mut params=p.clone();params.push(empty.clone());let mut o=Adafactor::new(params.clone(),&config()).unwrap();
 let before=state(&o);gradient(&p[0],0);empty.tensor().backward_with(&Tensor::zeros(&[0,3]));
 assert!(o.step().is_err());assert_eq!(state(&o),before);o.zero_grad();
 assert!(ferro_core::capture(||o.step()).is_err());assert_eq!(state(&o),before);
 let mut saved=o.snapshot();saved.iter_mut().find(|(n,_)|n=="t.1").unwrap().1=Tensor::scalar((1u32<<24) as f32);o.restore(&saved).unwrap();
 let before=state(&o);let values=p[0].tensor().to_vec();gradient(&p[0],1);gradient(&p[1],1);assert!(o.step().is_err());assert_eq!(state(&o),before);assert_eq!(p[0].tensor().to_vec(),values);
 o.zero_grad();p[1].set_trainable(false);gradient(&p[0],2);o.step().unwrap();assert_eq!(o.snapshot().iter().find(|(n,_)|n=="t.1").unwrap().1.item(),(1u32<<24)as f32);
}

#[test]
fn derived_factor_capacity_is_checked_for_zero_storage_parameters() {
 let huge=Param::new(Tensor::from_vec(vec![],&[1usize<<62,0]).unwrap());
 assert!(Adafactor::new(vec![huge],&config()).is_err());
}
