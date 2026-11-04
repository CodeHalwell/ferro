//! CPU float32 Muon with explicitly BF16-rounded Newton-Schulz iterations.
use super::{commit_buffers,validate_buffer_copies,validate_cpu_buffers,validate_cpu_f32_parameters,OptimizerState,StagedParameterValues};
use crate::{DType,Device,Error,Param,Result,Tensor};

fn invalid(message:&str)->Error {Error::Format{op:"muon",msg:message.into()}}
fn validate_config(c:&[f32])->Result<()> {
 // tag, lr, decay, momentum, nesterov, a, b, c, eps, iterations, adjustment.
 if c.len()!=11 || c[0]!=15. || c.iter().any(|v|!v.is_finite())
  || [1,2,3,8,9].iter().any(|&i|c[i]<0.) || ![0.,1.].contains(&c[4])
  || c[8]>f32::from_bits(0x7f7f0000) || c[9].fract()!=0. || c[9]>=100. || ![0.,1.,2.].contains(&c[10]) {
  return Err(invalid("invalid Muon configuration"));
 }
 Ok(())
}
fn validate_shapes(params:&[Param])->Result<()> {
 for p in params {
  let t=p.tensor();let shape=t.shape();
  if t.dtype()!=DType::F32 || shape.len()!=2 || shape[1]==0 {
   return Err(invalid("Muon requires float32 matrix parameters with nonzero columns"));
  }
  let rows=shape[0].min(shape[1]);
  for layout in [shape.to_vec(),vec![rows,rows]] {
   let n=crate::shape::checked_numel("muon",&layout)?;
   if n>isize::MAX as usize/4 {return Err(invalid("Muon work array exceeds addressable storage"));}
  }
 }
 Ok(())
}
fn bf16(value:f32)->f32 {
 if value.is_nan(){return f32::from_bits((value.to_bits()|0x0040_0000)&0xffff_0000);}
 let bits=value.to_bits();
 f32::from_bits(bits.wrapping_add(0x7fff+((bits>>16)&1))&0xffff_0000)
}
fn lerp(old:f32,new:f32,weight:f32)->f32 {if weight<0.5 {old+weight*(new-old)} else {new-(1.-weight)*(new-old)}}
fn orthogonalize(gradient:&[f32],rows:usize,cols:usize,config:&[f32])->Vec<f32> {
 let transposed=rows>cols;let (r,c)=if transposed {(cols,rows)} else {(rows,cols)};
 let mut x=vec![0.;gradient.len()];
 for i in 0..r {for j in 0..c {x[i*c+j]=bf16(gradient[if transposed {j*cols+i} else {i*cols+j}]);}}
 let norm=bf16(x.iter().map(|v|v*v).sum::<f32>().sqrt());let eps=bf16(config[8]);
 let denominator=if norm<eps {eps} else {norm};
 for value in &mut x {*value=bf16(*value/denominator);}
 for _ in 0..config[9] as usize {
  let mut gram=vec![0.;r*r];
  for i in 0..r {for j in 0..r {let mut sum=0.;for k in 0..c {sum+=x[i*c+k]*x[j*c+k];}gram[i*r+j]=bf16(sum);}}
  let mut update=vec![0.;r*r];
  for i in 0..r {for j in 0..r {let mut sum=0.;for k in 0..r {sum+=gram[i*r+k]*gram[k*r+j];}
   update[i*r+j]=bf16(config[6]*gram[i*r+j]+config[7]*sum);
  }}
  let mut next=vec![0.;x.len()];
  for i in 0..r {for j in 0..c {let mut sum=0.;for k in 0..r {sum+=update[i*r+k]*x[k*c+j];}
   next[i*c+j]=bf16(config[5]*x[i*c+j]+sum);
  }}
  x=next;
 }
 if transposed {let mut out=vec![0.;x.len()];for i in 0..r {for j in 0..c {out[j*cols+i]=x[i*c+j];}}out} else {x}
}

/// Parameter slots are shared; clones and snapshots own their momentum buffers.
pub struct Muon {params:Vec<Param>,config:Vec<f32>,initialized:Vec<bool>,buffers:Vec<Option<Tensor>>}
impl Clone for Muon {
 fn clone(&self)->Self {let mut out=self.clone_for_restore();out.buffers=self.buffers.iter().map(|t|t.as_ref().map(Tensor::owned_detach_copy)).collect();out}
}
impl Muon {
 pub fn new(params:Vec<Param>,config:&[f32])->Result<Self> {
  validate_config(config)?;let mut seen=std::collections::HashSet::new();
  let params:Vec<_>=params.into_iter().filter(|p|seen.insert(p.identity())).collect();
  validate_cpu_f32_parameters(&params)?;validate_shapes(&params)?;
  Ok(Self{initialized:vec![false;params.len()],buffers:vec![None;params.len()],params,config:config.to_vec()})
 }
 fn clone_for_restore(&self)->Self {Self{params:self.params.clone(),config:self.config.clone(),initialized:self.initialized.clone(),buffers:self.buffers.clone()}}
 pub fn zero_grad(&self){for p in &self.params {p.zero_grad();}}
 pub fn step(&mut self)->Result<()> {
  if crate::capture::is_recording(){return Err(Error::Unsupported{op:"muon",msg:"Muon requires eager CPU execution".into()});}
  super::validate_gradient_storage(&self.params)?;
  validate_cpu_f32_parameters(&self.params)?;validate_shapes(&self.params)?;validate_cpu_buffers(&self.buffers)?;
  for(i,p)in self.params.iter().enumerate(){
   if self.buffers[i].as_ref().is_some_and(|t|t.shape()!=p.tensor().shape()){return Err(invalid("parameter shape changed after state initialization"));}
   if let Some(g)=p.grad(){if g.device()!=Device::Cpu || g.dtype()!=DType::F32 || g.shape()!=p.tensor().shape(){return Err(invalid("Muon gradients must match CPU float32 parameters"));}}
  }
  let mut pending=vec![];let mut shared_values=StagedParameterValues::default();
  for(i,p)in self.params.iter().enumerate(){
   let Some(gradient)=p.grad() else {continue};let parameter=p.tensor();let c=&self.config;
   let mut momentum=self.buffers[i].as_ref().map(Tensor::to_vec).unwrap_or_else(||vec![0.;parameter.numel()]);
   let grads=gradient.to_vec();let weight=(1.-c[3] as f64) as f32;
   for(old,&g)in momentum.iter_mut().zip(&grads){*old=lerp(*old,g,weight);}
   let update=if c[4]==1.{grads.iter().zip(&momentum).map(|(&g,&m)|lerp(g,m,c[3])).collect()}else{momentum.clone()};
   let (rows,cols)=(parameter.shape()[0],parameter.shape()[1]);
   let update=orthogonalize(&update,rows,cols,c);
   let ratio=if c[10]==2.{0.2*(rows.max(cols) as f64).sqrt()}else{1f64.max(rows as f64/cols as f64).sqrt()};
   let decay=(1.-c[1] as f64*c[2] as f64) as f32;let scale=(-c[1] as f64*ratio) as f32;
   let current=shared_values.read(&parameter,||parameter.to_vec());
   let values:Vec<f32>=current.iter().zip(&update).map(|(&v,&u)|v*decay+u*scale).collect();
   let next=Tensor::from_vec(values,parameter.shape())?;shared_values.stage(&parameter,&next);
   pending.push((i,parameter.clone(),next,Some(Tensor::from_vec(momentum,parameter.shape())?)));
  }
  for(i,parameter,next,mut buffer)in pending {
   commit_buffers(std::slice::from_ref(&self.buffers[i]),std::slice::from_mut(&mut buffer)).expect("validated Muon buffer");
   crate::inplace::raw_copy_("muon",&parameter,&next).expect("validated Muon parameter");
   self.buffers[i]=buffer;self.initialized[i]=true;self.params[i].zero_grad();
  }
  Ok(())
 }
 fn restore_staged(&mut self,state:&[(String,Tensor)])->Result<()> {
  if state.len()!=self.params.len()*2 || state.iter().map(|(k,_)|k).collect::<std::collections::HashSet<_>>().len()!=state.len(){return Err(invalid("Muon state keys mismatch"));}
  for i in 0..self.params.len(){
   let find=|name:&str|state.iter().find(|(k,_)|k==&format!("{name}.{i}")).map(|(_,v)|v).ok_or_else(||invalid("missing Muon state key"));
   let flag=find("initialized")?;
   if flag.dtype()!=DType::F32 || flag.device()!=Device::Cpu || flag.shape()!=[] || ![0.,1.].contains(&flag.item()){return Err(invalid("invalid Muon initialized flag"));}
   let value=find("momentum_buffer")?;let parameter=self.params[i].tensor();
   if value.dtype()!=DType::F32 || value.device()!=Device::Cpu || value.shape()!=[parameter.numel()]{return Err(invalid("Muon momentum shape/dtype/device mismatch"));}
   if flag.item()==0. && value.to_vec().iter().any(|v|*v!=0.){return Err(invalid("uninitialized Muon momentum must be zero"));}
   self.initialized[i]=flag.item()==1.;self.buffers[i]=Some(value.owned_detach_copy().reshape(parameter.shape())?);
  }
  Ok(())
 }
}
impl OptimizerState for Muon {
 fn expected_restore_keys(&self,c:&[f32])->Result<Vec<String>>{validate_config(c)?;Ok((0..self.params.len()).flat_map(|i|["initialized","momentum_buffer"].map(|name|format!("{name}.{i}"))).collect())}
 fn state_parameters(&self)->Option<Vec<Param>>{Some(self.params.clone())}
 fn state_buffers(&self)->Option<Vec<Tensor>>{Some(self.buffers.iter().flatten().cloned().collect())}
 fn configuration(&self)->Vec<f32>{self.config.clone()}
 fn restore_configuration(&mut self,c:&[f32])->Result<()>{validate_config(c)?;self.config=c.to_vec();Ok(())}
 fn snapshot(&self)->Vec<(String,Tensor)>{
  let mut out=vec![];
  for i in 0..self.params.len(){
   out.push((format!("initialized.{i}"),Tensor::scalar(self.initialized[i] as u8 as f32)));
   let n=self.params[i].tensor().numel();let value=self.buffers[i].as_ref().map(Tensor::owned_detach_copy).unwrap_or_else(||Tensor::zeros(&[n]));
   out.push((format!("momentum_buffer.{i}"),value.reshape(&[value.numel()]).expect("Muon snapshot shape")));
  }
  out
 }
 fn prepare_cpu_restore<'a>(&'a mut self,c:&[f32],state:&[(String,Tensor)])->Result<Box<dyn FnOnce()+'a>>{
  validate_cpu_f32_parameters(&self.params)?;validate_shapes(&self.params)?;validate_cpu_buffers(&self.buffers)?;
  let mut staged=self.clone_for_restore();staged.restore_configuration(c)?;staged.restore_staged(state)?;
  validate_buffer_copies(&self.buffers,&staged.buffers)?;
  Ok(Box::new(move||{commit_buffers(&self.buffers,&mut staged.buffers).expect("validated Muon buffers");*self=staged;}))
 }
 fn restore(&mut self,state:&[(String,Tensor)])->Result<()>{let c=self.config.clone();self.prepare_cpu_restore(&c,state)?();Ok(())}
}

#[cfg(test)]
mod tests {
 use super::bf16;
 #[test]
 fn bf16_rounding_preserves_signed_zero_and_uses_even_ties(){
  for (input,expected) in [(0x0000_0000,0x0000_0000),(0x8000_0000,0x8000_0000),
   (0x3f80_8000,0x3f80_0000),(0x3f81_8000,0x3f82_0000),(0xbf80_8000,0xbf80_0000),(0xbf81_8000,0xbf82_0000),
   (0x0000_8000,0x0000_0000),(0x0001_8000,0x0002_0000),(0x7f7f_ffff,0x7f80_0000),(0x7f80_0000,0x7f80_0000)]{
   assert_eq!(bf16(f32::from_bits(input)).to_bits(),expected);
  }
  for bits in [0x7f80_0001,0x7fc0_0000,0xffff_ffff]{let value=bf16(f32::from_bits(bits));assert!(value.is_nan());assert_eq!(value.to_bits()&0xffff,0);}
 }
}
