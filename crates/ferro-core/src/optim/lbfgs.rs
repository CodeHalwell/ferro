//! Eager CPU f32 L-BFGS with fixed or Strong-Wolfe steps and a loss closure.
//! History lives in owned host arrays; checkpoints are valid between complete
//! calls. Missing gradients are zero coordinates, not skipped parameter slots.
use super::{validate_cpu_f32_parameters, OptimizerState, StagedParameterValues};
use crate::{DType, Device, Error, Param, Result, Tensor};
mod line_search;
const MAX_COUNT: u32 = 1 << 24;
fn invalid(message: &str) -> Error { Error::Format { op: "lbfgs", msg: message.into() } }
fn config_valid(c: &[f32]) -> Result<()> {
    // tag, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size
    if ![7,8].contains(&c.len()) || (c.len()==8 && c[7]!=1.) || c[0] != 13.0 || c.iter().any(|x| !x.is_finite() || *x < 0.0)
        || [2,3,6].iter().any(|&i| c[i] < 1.0 || c[i].fract() != 0.0 || c[i] > MAX_COUNT as f32) {
        return Err(invalid("invalid LBFGS configuration"));
    }
    Ok(())
}
fn dot(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(a,b)| a*b).sum() }
fn max_abs(a: &[f32]) -> f32 { a.iter().map(|x| x.abs()).fold(0.0, f32::max) }

/// One coupled parameter group, CPU f32, no capture. Optional Strong-Wolfe
/// line search evaluates from a preserved baseline on every trial. Cloning
/// shares parameter slots but owns independent history. Errors from a closure
/// after updates can leave partial progress; discard that run rather than
/// treating an unsuccessful call as an atomic optimization step.
#[derive(Clone)]
pub struct Lbfgs {
    params: Vec<Param>, shapes: Vec<Vec<usize>>, config: Vec<f32>, n: usize,
    func_evals: u32, n_iter: u32, details: bool, alpha_present: bool,
    d: Vec<f32>, prev_grad: Vec<f32>, t: f32, h_diag: f32, prev_loss: f32,
    old_dirs: Vec<Vec<f32>>, old_stps: Vec<Vec<f32>>, rho: Vec<f32>, alpha: Vec<Option<f32>>,
}
impl Lbfgs {
    pub fn new(params: Vec<Param>, config: &[f32]) -> Result<Self> {
        config_valid(config)?;
        let mut seen = std::collections::HashSet::new();
        let params: Vec<_> = params.into_iter().filter(|p| seen.insert(p.identity())).collect();
        validate_cpu_f32_parameters(&params)?;
        let n = params.iter().try_fold(0usize, |n,p| n.checked_add(p.tensor().numel())).ok_or_else(||invalid("parameter capacity overflow"))?;
        if n == 0 { return Err(invalid("LBFGS requires at least one parameter element")); }
        let capacity = crate::shape::checked_numel("lbfgs", &[config[6] as usize,n])?;
        if capacity > isize::MAX as usize / 4 { return Err(invalid("history capacity exceeds addressable storage")); }
        let shapes = params.iter().map(|p|p.tensor().shape().to_vec()).collect();
        let this = Self { params, shapes, config:config.to_vec(), n, func_evals:0, n_iter:0, details:false,
            alpha_present:false, d:vec![0.;n], prev_grad:vec![0.;n], t:0., h_diag:0., prev_loss:0.,
            old_dirs:vec![], old_stps:vec![], rho:vec![], alpha:vec![None;config[6] as usize] };
        this.validate_parameters()?;
        Ok(this)
    }
    fn validate_parameters(&self) -> Result<()> {
        validate_cpu_f32_parameters(&self.params)?;
        if self.params.iter().zip(&self.shapes).any(|(p,s)| !p.is_trainable() || p.tensor().shape() != s) {
            return Err(invalid("LBFGS requires trainable parameters with unchanged shapes"));
        }
        Ok(())
    }
    pub fn zero_grad(&self) { for p in &self.params { p.zero_grad(); } }
    fn gather(&self) -> Result<Vec<f32>> {
        self.validate_parameters()?;
        super::validate_gradient_storage(&self.params)?;
        let mut result = Vec::with_capacity(self.n);
        for p in &self.params {
            match p.grad() {
                Some(g) => {
                    if g.device()!=Device::Cpu || g.dtype()!=DType::F32 || g.shape()!=p.tensor().shape() { return Err(invalid("gradient shape/dtype/device mismatch")); }
                    result.extend(g.to_vec());
                }, None => result.resize(result.len()+p.tensor().numel(),0.),
            }
        }
        if result.iter().any(|x| !x.is_finite()) { return Err(invalid("LBFGS requires finite gradients")); }
        Ok(result)
    }
    fn evaluate<F: FnMut()->Result<f32>>(&mut self, closure: &mut F) -> Result<(f32,Vec<f32>)> {
        if self.func_evals == MAX_COUNT { return Err(invalid("function evaluation counter exceeds exact checkpoint range")); }
        let loss = closure()?;
        if !loss.is_finite() { return Err(invalid("LBFGS requires a finite scalar closure loss")); }
        let grad = self.gather()?;
        self.func_evals += 1;
        Ok((loss,grad))
    }
    fn write_from(&self, baseline: &[Vec<f32>], direction: &[f32], step: Option<f32>) -> Result<()> {
        self.validate_parameters()?;
        let mut pending=Vec::new(); let mut offset=0; let mut staged_values=StagedParameterValues::default();
        for (p,base) in self.params.iter().zip(baseline) {
            let target=p.tensor();
            let values=match step {
                Some(t)=>staged_values.read(&target,||base.clone()).iter().zip(&direction[offset..]).map(|(p,d)|*p+t**d).collect(),
                None=>base.clone(),
            };
            offset+=target.numel();
            if values.iter().any(|v|!v.is_finite()) {return Err(invalid("LBFGS update is nonfinite"));}
            let next=Tensor::from_vec(values,target.shape())?;
            staged_values.stage(&target,&next);
            pending.push((target,next));
        }
        for (target,value) in pending {crate::inplace::raw_copy_("lbfgs",&target,&value)?;}
        Ok(())
    }
    fn add_step(&self) -> Result<()> {
        let baseline:Vec<_>=self.params.iter().map(|p|p.tensor().to_vec()).collect();
        self.write_from(&baseline,&self.d,Some(self.t))
    }
    fn evaluate_trial<F:FnMut()->Result<f32>>(&mut self, baseline:&[Vec<f32>], direction:&[f32], step:f32, closure:&mut F)->Result<(f32,Vec<f32>)> {
        self.write_from(baseline,direction,Some(step))?;
        let outcome=self.evaluate(closure);
        // Copy the baseline exactly (including signed zeros), also on closure
        // failure. Earlier iterations/history may still contain partial work.
        self.write_from(baseline,direction,None)?;
        outcome
    }
    /// Return the first loss value. The binding returns the original loss tensor.
    pub fn step<F: FnMut()->Result<f32>>(&mut self, mut closure: F) -> Result<f32> {
        if crate::capture::is_recording() { return Err(Error::Unsupported { op:"lbfgs",msg:"LBFGS requires eager execution".into() }); }
        super::validate_gradient_storage(&self.params)?;
        self.validate_parameters()?;
        let (mut loss,mut grad) = self.evaluate(&mut closure)?;
        let original = loss; let mut current_evals = 1;
        let mut optimal = max_abs(&grad) <= self.config[4];
        if optimal { self.zero_grad(); return Ok(original); }
        for iteration in 1..=self.config[2] as u32 {
            if self.n_iter == MAX_COUNT { return Err(invalid("iteration counter exceeds exact checkpoint range")); }
            self.n_iter += 1;
            if self.n_iter == 1 {
                self.d = grad.iter().map(|g| -g).collect(); self.h_diag=1.;
                self.old_dirs.clear(); self.old_stps.clear(); self.rho.clear();
            } else {
                let y: Vec<f32> = grad.iter().zip(&self.prev_grad).map(|(a,b)|a-b).collect();
                let s: Vec<f32> = self.d.iter().map(|d|d*self.t).collect();
                let ys = dot(&y,&s);
                if ys > 1e-10 {
                    if self.old_dirs.len() == self.config[6] as usize {
                        self.old_dirs.remove(0); self.old_stps.remove(0); self.rho.remove(0);
                    }
                    self.h_diag = ys / dot(&y,&y);
                    self.old_dirs.push(y); self.old_stps.push(s); self.rho.push(1./ys);
                }
                self.alpha_present=true;
                let count=self.old_dirs.len(); let mut q:Vec<f32>=grad.iter().map(|g|-g).collect();
                for i in (0..count).rev() {
                    let alpha = dot(&self.old_stps[i],&q)*self.rho[i]; self.alpha[i]=Some(alpha);
                    for (q,y) in q.iter_mut().zip(&self.old_dirs[i]) { *q -= alpha*y; }
                }
                self.d = q.iter().map(|q|q*self.h_diag).collect();
                for i in 0..count {
                    let beta=dot(&self.old_dirs[i],&self.d)*self.rho[i]; let coefficient=self.alpha[i].unwrap()-beta;
                    for (d,s) in self.d.iter_mut().zip(&self.old_stps[i]) { *d += coefficient*s; }
                }
            }
            self.prev_grad.clone_from(&grad); self.prev_loss=loss; self.details=true;
            self.t=if self.n_iter==1 { (1. / grad.iter().map(|g|g.abs()).sum::<f32>()).min(1.)*self.config[1] } else {self.config[1]};
            if self.d.iter().any(|v|!v.is_finite()) || !self.h_diag.is_finite() || !self.t.is_finite() { return Err(invalid("nonfinite LBFGS history")); }
            let slope=dot(&grad,&self.d);
            if slope > -self.config[5] { break; }
            if self.config.len()==8 {
                let baseline:Vec<_>=self.params.iter().map(|p|p.tensor().to_vec()).collect();
                let direction=self.d.clone(); let previous_evals=self.func_evals;
                let remaining=(self.config[3] as u32).saturating_sub(current_evals);
                let accepted=line_search::search(self.t,&direction,loss,&grad,slope,remaining,
                    |t|self.evaluate_trial(&baseline,&direction,t,&mut closure))?;
                self.t=accepted.t;loss=accepted.loss;grad=accepted.grad;
                self.add_step()?;
                current_evals+=self.func_evals-previous_evals;
                optimal=max_abs(&grad)<=self.config[4];
            } else {
                self.add_step()?;
                if iteration != self.config[2] as u32 {
                    (loss,grad)=self.evaluate(&mut closure)?; current_evals+=1;
                    optimal=max_abs(&grad)<=self.config[4];
                }
            }
            if iteration==self.config[2] as u32 || current_evals>=self.config[3] as u32 || optimal
                || max_abs(&self.d)*self.t<=self.config[5] || (loss-self.prev_loss).abs()<self.config[5] { break; }
        }
        self.zero_grad(); Ok(original)
    }
    fn restore_staged(&mut self, tensors:&[(String,Tensor)]) -> Result<()> {
        let expected=self.snapshot();
        if tensors.len()!=expected.len() || tensors.iter().map(|(n,_)|n).collect::<std::collections::HashSet<_>>().len()!=tensors.len() {return Err(invalid("LBFGS state keys mismatch"));}
        let mut data=std::collections::HashMap::new();
        for (name,expected) in expected {
            let t=&tensors.iter().find(|(n,_)|n==&name).ok_or_else(||invalid("missing LBFGS state"))?.1;
            if t.dtype()!=DType::F32 || t.device()!=Device::Cpu || t.shape()!=expected.shape() {return Err(invalid("LBFGS state shape/dtype/device mismatch"));}
            let values=t.to_vec(); if values.iter().any(|x|!x.is_finite()) {return Err(invalid("nonfinite LBFGS state"));}
            data.insert(name,values);
        }
        let header=&data["header"]; let scalars=&data["scalars"];
        if header.iter().any(|x|*x<0. || x.fract()!=0. || *x>MAX_COUNT as f32)
            || ![0.,1.].contains(&header[2]) || ![0.,1.].contains(&header[3]) || header[4]>self.config[6]
            || (header[1]>0.)!=(header[2]==1.) || header[1]>header[0] || (header[3]==1.)!=(header[1]>1.)
            || scalars[0]<0. || scalars[1]<0. {return Err(invalid("invalid LBFGS counters or flags"));}
        let count=header[4] as usize;
        if count>0 && (header[1]<2. || scalars[1]<=0.) {return Err(invalid("invalid LBFGS history count"));}
        for i in 0..self.config[6] as usize {
            let valid=data["alpha_valid"][i];
            if ![0.,1.].contains(&valid) || (valid==1.)!=(i<count) || (i<count && data["rho"][i]<=0.)
                || (i>=count && (data["rho"][i]!=0. || data["alpha"][i]!=0. || data["old_dirs"][i*self.n..(i+1)*self.n].iter().chain(&data["old_stps"][i*self.n..(i+1)*self.n]).any(|x|*x!=0.))) {return Err(invalid("invalid LBFGS history padding or coefficient"));}
        }
        if header[2]==0. && (count!=0 || scalars.iter().chain(&data["d"]).chain(&data["prev_grad"]).any(|v|*v!=0.)) {return Err(invalid("unexpected uninitialized LBFGS state"));}
        self.func_evals=header[0] as u32; self.n_iter=header[1] as u32; self.details=header[2]==1.;self.alpha_present=header[3]==1.;
        self.t=scalars[0];self.h_diag=scalars[1];self.prev_loss=scalars[2];
        self.d=data["d"].clone();self.prev_grad=data["prev_grad"].clone();
        self.old_dirs=data["old_dirs"].chunks(self.n).take(count).map(|v|v.to_vec()).collect();
        self.old_stps=data["old_stps"].chunks(self.n).take(count).map(|v|v.to_vec()).collect();self.rho=data["rho"][..count].to_vec();
        self.alpha=data["alpha"].iter().enumerate().map(|(i,a)|if i<count {Some(*a)} else {None}).collect();
        Ok(())
    }
}
impl OptimizerState for Lbfgs {
    fn state_parameters(&self)->Option<Vec<Param>> {Some(self.params.clone())}
    fn state_buffers(&self)->Option<Vec<Tensor>> {Some(vec![])}
    fn configuration(&self)->Vec<f32> {self.config.clone()}
    fn restore_configuration(&mut self,c:&[f32])->Result<()> {
        config_valid(c)?;
        if c[6]!=self.config[6] {return Err(invalid("LBFGS history_size must match the constructed optimizer"));}
        self.config=c.to_vec();Ok(())
    }
    fn snapshot(&self)->Vec<(String,Tensor)> {
        let h=self.config[6] as usize;let count=self.old_dirs.len();
        let mut dirs=vec![0.;h*self.n];let mut stps=dirs.clone();let mut rho=vec![0.;h];
        for i in 0..count {dirs[i*self.n..(i+1)*self.n].copy_from_slice(&self.old_dirs[i]);stps[i*self.n..(i+1)*self.n].copy_from_slice(&self.old_stps[i]);rho[i]=self.rho[i];}
        [("header",vec![self.func_evals as f32,self.n_iter as f32,self.details as u8 as f32,self.alpha_present as u8 as f32,count as f32],vec![5]),
         ("scalars",vec![self.t,self.h_diag,self.prev_loss],vec![3]),("d",self.d.clone(),vec![self.n]),("prev_grad",self.prev_grad.clone(),vec![self.n]),
         ("old_dirs",dirs,vec![h,self.n]),("old_stps",stps,vec![h,self.n]),("rho",rho,vec![h]),
         ("alpha",self.alpha.iter().map(|a|a.unwrap_or(0.)).collect(),vec![h]),("alpha_valid",self.alpha.iter().map(|a|a.is_some() as u8 as f32).collect(),vec![h])]
            .into_iter().map(|(name,data,shape)|(name.into(),Tensor::from_vec(data,&shape).expect("validated LBFGS capacity"))).collect()
    }
    fn prepare_cpu_restore<'a>(&'a mut self,config:&[f32],state:&[(String,Tensor)])->Result<Box<dyn FnOnce()+'a>> {
        self.validate_parameters()?;
        let mut staged=self.clone();staged.restore_configuration(config)?;staged.restore_staged(state)?;
        Ok(Box::new(move || {*self=staged;}))
    }
    fn restore(&mut self,state:&[(String,Tensor)])->Result<()> {let config=self.config.clone();self.prepare_cpu_restore(&config,state)?();Ok(())}
}
