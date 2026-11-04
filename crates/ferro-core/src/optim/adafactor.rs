//! CPU f32 Adafactor, retaining row/column factors for matrix parameters.
use super::{commit_buffers,validate_buffer_copies,validate_cpu_buffers,validate_cpu_f32_parameters,OptimizerState,StagedParameterValues};
use crate::{DType,Device,Error,Param,Result,Tensor};
const NAMES:[&str;3]=["row_var","col_var","variance"];
const MAX_STEP:u32=1<<24;
fn invalid(message:&str)->Error {Error::Format{op:"adafactor",msg:message.into()}}
fn validate_config(c:&[f32])->Result<()> {
 // tag, lr, beta2_decay, eps1 (-1 means float32 epsilon), eps2, d, decay, maximize.
 if c.len()!=8 || c[0]!=14. || c.iter().any(|v|!v.is_finite()) || c[1]<0. || c[2]>0.
  || (c[3]<0. && c[3]!=-1.) || c[4]<0. || c[5]<1. || c[6]<0. || ![0.,1.].contains(&c[7]) {return Err(invalid("invalid Adafactor configuration"));}
 Ok(())
}
fn lerp(old:f32,new:f32,weight:f32)->f32 {if weight<0.5 {old+weight*(new-old)} else {new-(1.-weight)*(new-old)}}
fn norm(values:&[f32])->f32 {values.iter().map(|v|v*v).sum::<f32>().sqrt()}
fn shape(parameter:&Tensor,field:usize)->Vec<usize> {
 let mut result=parameter.shape().to_vec();let rank=result.len();
 if rank>=2 {match field {0=>result[rank-1]=1,1=>result[rank-2]=1,_=>return vec![0]};result}
 else if field==2 {result} else {vec![0]}
}
fn validate_factor_capacity(params:&[Param])->Result<()> {
 for p in params {for field in 0..3 {
  let layout:Vec<_>=shape(&p.tensor(),field).into_iter().map(|d|d.max(1)).collect();
  let n=crate::shape::checked_numel("adafactor",&layout)?;
  if n>isize::MAX as usize/4 {return Err(invalid("derived Adafactor factor exceeds addressable storage"));}
 }}
 Ok(())
}
/// Clones share parameter slots and own independent second-moment factors.
pub struct Adafactor {params:Vec<Param>,config:Vec<f32>,steps:Vec<u32>,buffers:Vec<Vec<Option<Tensor>>>}
impl Clone for Adafactor {
 fn clone(&self)->Self {let mut result=self.clone_for_restore();result.buffers=self.buffers.iter().map(|v|v.iter().map(|t|t.as_ref().map(Tensor::owned_detach_copy)).collect()).collect();result}
}
impl Adafactor {
 pub fn new(params:Vec<Param>,config:&[f32])->Result<Self> {
  validate_config(config)?;let mut seen=std::collections::HashSet::new();let params:Vec<_>=params.into_iter().filter(|p|seen.insert(p.identity())).collect();validate_cpu_f32_parameters(&params)?;
  validate_factor_capacity(&params)?;
  Ok(Self{steps:vec![0;params.len()],buffers:vec![vec![None;3];params.len()],params,config:config.to_vec()})
 }
 fn clone_for_restore(&self)->Self {Self{params:self.params.clone(),config:self.config.clone(),steps:self.steps.clone(),buffers:self.buffers.clone()}}
 pub fn zero_grad(&self){for p in &self.params {p.zero_grad();}}
 pub fn step(&mut self)->Result<()> {
  if crate::capture::is_recording(){return Err(Error::Unsupported{op:"adafactor",msg:"Adafactor requires eager CPU execution".into()});}
  super::validate_gradient_storage(&self.params)?;
  validate_cpu_f32_parameters(&self.params)?;
  validate_factor_capacity(&self.params)?;
  for (i,p) in self.params.iter().enumerate() {
   validate_cpu_buffers(&self.buffers[i])?;
   for (field,t) in self.buffers[i].iter().enumerate(){if let Some(t)=t {if t.shape()!=shape(&p.tensor(),field){return Err(invalid("parameter shape changed after state initialization"));}}}
   if let Some(g)=p.grad(){if g.dtype()!=DType::F32 || g.device()!=Device::Cpu || g.shape()!=p.tensor().shape() || g.numel()==0 {return Err(invalid("Adafactor active gradients must match nonempty CPU f32 parameters"));}
    if self.steps[i]>=MAX_STEP{return Err(invalid("step exceeds exact checkpoint counter range"));}}
  }
  let mut pending=vec![];let mut staged_values=StagedParameterValues::default();
  for (i,p) in self.params.iter().enumerate() {
   let Some(gradient)=p.grad() else {continue};let parameter=p.tensor();let n=parameter.numel();let c=&self.config;
   let mut values=staged_values.read(&parameter,||parameter.to_vec());let mut grad=gradient.to_vec();if c[7]==1.{for v in &mut grad{*v = -*v;}}
   let step=self.steps[i]+1;let beta=(step as f64).powf(c[2] as f64) as f32;
   let relative=(c[1] as f64).min(1./(step as f64).sqrt());
   let alpha=(c[4] as f64).max(norm(&values) as f64/(n as f64).sqrt())*relative;
   let eps=if c[3]==-1. {f32::EPSILON} else {c[3]};let eps_square=(eps as f64*eps as f64) as f32;
   let mut factors:Vec<Vec<f32>>=(0..3).map(|f|self.buffers[i][f].as_ref().map(Tensor::to_vec).unwrap_or_else(||vec![0.;shape(&parameter,f).iter().product()])).collect();
   let mut update=vec![0.;n];let rank=parameter.shape().len();
   if rank>=2 {
    let rows=parameter.shape()[rank-2];let cols=parameter.shape()[rank-1];let batches=n/(rows*cols);
    for b in 0..batches {
     let start=b*rows*cols;
     for r in 0..rows {
      let magnitude=norm(&grad[start+r*cols..start+(r+1)*cols]);
      factors[0][b*rows+r]=lerp(factors[0][b*rows+r],magnitude*magnitude/cols as f32,beta);
     }
     for col in 0..cols {
      let magnitude=(0..rows).map(|r|grad[start+r*cols+col].powi(2)).sum::<f32>().sqrt();
      factors[1][b*cols+col]=lerp(factors[1][b*cols+col],magnitude*magnitude/rows as f32,beta);
     }
     let mean=factors[0][b*rows..(b+1)*rows].iter().sum::<f32>()/rows as f32;let divisor=if mean<eps {eps} else {mean};
     for r in 0..rows {for col in 0..cols {
      let j=start+r*cols+col;let estimate=factors[0][b*rows+r]*factors[1][b*cols+col]/divisor;
      let bounded=if estimate<eps_square {eps_square} else {estimate};update[j]=grad[j]/bounded.sqrt();
     }}
    }
   } else {
    for j in 0..n {factors[2][j]=lerp(factors[2][j],grad[j]*grad[j],beta);let variance=factors[2][j];let bounded=if variance<eps_square {eps_square} else {variance};update[j]=grad[j]/bounded.sqrt();}
   }
   let denominator=1f64.max(norm(&update) as f64/((n as f64).sqrt()*c[5] as f64));let scale=(-alpha/denominator) as f32;
   let decay=(1.-c[1] as f64*c[6] as f64) as f32;
   for j in 0..n {if c[6]!=0.{values[j]*=decay;}values[j]+=scale*update[j];}
   let next=Tensor::from_vec(values,parameter.shape())?;
   staged_values.stage(&parameter,&next);
   let state=factors.into_iter().enumerate().map(|(f,v)|Tensor::from_vec(v,&shape(&parameter,f)).map(Some)).collect::<Result<Vec<_>>>()?;
   pending.push((i,parameter,next,state));
  }
  for (i,parameter,next,mut state) in pending {
   commit_buffers(&self.buffers[i],&mut state).expect("validated Adafactor buffers");
   crate::inplace::raw_copy_("adafactor",&parameter,&next).expect("validated CPU parameter");
   self.buffers[i]=state;self.steps[i]+=1;self.params[i].zero_grad();
  }
  Ok(())
 }
 fn restore_staged(&mut self,state:&[(String,Tensor)])->Result<()> {
  if state.len()!=self.params.len()*4 || state.iter().map(|(k,_)|k).collect::<std::collections::HashSet<_>>().len()!=state.len(){return Err(invalid("Adafactor state keys mismatch"));}
  for i in 0..self.params.len(){
   let find=|name:&str|state.iter().find(|(k,_)|k==&format!("{name}.{i}")).map(|(_,t)|t).ok_or_else(||invalid("missing Adafactor state key"));
   let counter=find("t")?;
   if counter.dtype()!=DType::F32 || counter.device()!=Device::Cpu || counter.shape()!=[]{return Err(invalid("step must be scalar CPU f32"));}
   let step=counter.item();if !step.is_finite() || step<0. || step.fract()!=0. || step>MAX_STEP as f32{return Err(invalid("invalid Adafactor step"));}self.steps[i]=step as u32;
   for (f,name) in NAMES.iter().enumerate(){let value=find(name)?;let target_shape=shape(&self.params[i].tensor(),f);let n:usize=target_shape.iter().product();
    if value.dtype()!=DType::F32 || value.device()!=Device::Cpu || value.shape()!=[n]{return Err(invalid("Adafactor factor shape/dtype/device mismatch"));}
    self.buffers[i][f]=Some(value.owned_detach_copy().reshape(&target_shape)?);
   }
  }
  Ok(())
 }
}
impl OptimizerState for Adafactor {
 fn expected_restore_keys(&self,c:&[f32])->Result<Vec<String>>{
  validate_config(c)?;
  Ok((0..self.params.len()).flat_map(|i|["t","row_var","col_var","variance"].map(|name|format!("{name}.{i}"))).collect())
 }
 fn state_parameters(&self)->Option<Vec<Param>>{Some(self.params.clone())}
 fn state_buffers(&self)->Option<Vec<Tensor>>{Some(self.buffers.iter().flatten().flatten().cloned().collect())}
 fn configuration(&self)->Vec<f32>{self.config.clone()}
 fn restore_configuration(&mut self,c:&[f32])->Result<()>{validate_config(c)?;self.config=c.to_vec();Ok(())}
 fn snapshot(&self)->Vec<(String,Tensor)>{let mut out=vec![];for i in 0..self.params.len(){out.push((format!("t.{i}"),Tensor::scalar(self.steps[i] as f32)));
  for (f,name) in NAMES.iter().enumerate(){let n:usize=shape(&self.params[i].tensor(),f).iter().product();let value=self.buffers[i][f].as_ref().map(Tensor::owned_detach_copy).unwrap_or_else(||Tensor::zeros(&[n]));out.push((format!("{name}.{i}"),value.reshape(&[n]).expect("Adafactor snapshot shape")));}}
  out
 }
 fn prepare_cpu_restore<'a>(&'a mut self,c:&[f32],state:&[(String,Tensor)])->Result<Box<dyn FnOnce()+'a>>{
  validate_cpu_f32_parameters(&self.params)?;for b in &self.buffers{validate_cpu_buffers(b)?;}
  validate_factor_capacity(&self.params)?;
  let mut staged=self.clone_for_restore();staged.restore_configuration(c)?;staged.restore_staged(state)?;
  for(old,new)in self.buffers.iter().zip(&staged.buffers){validate_buffer_copies(old,new)?;}
  Ok(Box::new(move||{for(old,new)in self.buffers.iter().zip(&mut staged.buffers){commit_buffers(old,new).expect("validated Adafactor buffers");}*self=staged;}))
 }
 fn restore(&mut self,state:&[(String,Tensor)])->Result<()>{let c=self.config.clone();self.prepare_cpu_restore(&c,state)?();Ok(())}
}
