//! CPU Double Adam arithmetic. Float32 slots still use the original kernels.
use super::*;
use crate::DType;

fn invalid(message: &str) -> Error { Error::Format { op: "adam", msg: message.into() } }

pub(super) fn validate_real_parameters(params: &[Param]) -> Result<()> {
    if params.iter().any(|p| { let t = p.tensor(); t.dtype() != DType::F32 && !(t.dtype() == DType::F64 && t.device() == Device::Cpu) }) {
        return Err(invalid("Adam requires float32 parameters or CPU float64 parameters"));
    }
    if params.iter().any(|p| p.tensor().dtype() == DType::F64) && params.iter().any(|p| p.tensor().device() != Device::Cpu) {
        return Err(invalid("mixed float64 Adam groups require CPU parameters"));
    }
    Ok(())
}

fn encode(kind: f32, values: [f64;6], amsgrad: bool, maximize: bool, clip: bool) -> Vec<f32> {
    let mut result = vec![kind,64.,amsgrad as u8 as f32,maximize as u8 as f32,clip as u8 as f32];
    for value in values { for shift in [0,16,32,48] { result.push(((value.to_bits() >> shift) & 65535) as f32); } }
    result
}

fn validate_scalars(values: [f64;6], params: &[Param]) -> Result<()> {
    if values.iter().any(|x| !x.is_finite() || *x < 0.) || values[1] >= 1. || values[2] >= 1. {
        return Err(invalid("invalid exact Adam scalars"));
    }
    if params.iter().any(|p| p.tensor().dtype() == DType::F32)
        && (values.iter().any(|&x| !(x as f32).is_finite())
            || values[1] as f32 >= 1. || values[2] as f32 >= 1.
            || (values[3] != 0. && values[3] as f32 == 0.)) {
        return Err(invalid("Adam scalars for float32 parameters must fit float32"));
    }
    Ok(())
}

fn state_buffer(tensors: &[(String,Tensor)], name: &str, parameter: &Tensor) -> Result<Tensor> {
    let value = &tensors.iter().find(|(n,_)| n == name).ok_or_else(|| invalid("missing Adam state buffer"))?.1;
    if value.dtype() != parameter.dtype() || value.device() != Device::Cpu || value.shape() != [parameter.numel()]
        || !value.is_contiguous() || value.0.offset != 0 || value.storage_len() != value.numel() {
        return Err(invalid("Adam state buffers must be whole contiguous CPU arrays with the parameter dtype"));
    }
    Ok(value.owned_detach_copy().reshape(parameter.shape())?)
}

fn steps(value: &Tensor, expected_shape: &[usize]) -> Result<Vec<u32>> {
    if value.device() != Device::Cpu || !matches!(value.dtype(),DType::F32|DType::F64)
        || value.shape() != expected_shape || !value.is_contiguous() || value.0.offset != 0 || value.storage_len() != value.numel() {
        return Err(invalid("Adam counters must be whole CPU floating arrays of the expected shape"));
    }
    let values = value.to_vec_f64();
    if values.iter().any(|x| !x.is_finite() || *x < 0. || *x > u32::MAX as f64 || x.fract() != 0.) {
        return Err(invalid("Adam counter is outside the unsigned 32-bit integer range"));
    }
    Ok(values.into_iter().map(|x| x as u32).collect())
}

macro_rules! precise_adam {
    ($ty:ident, $kind:expr, $decoupled:expr) => {
        impl $ty {
            pub fn lr_f64(&self) -> f64 { self.f64_values()[0] }
            pub fn set_lr_f64(&mut self, lr: f64) -> Result<()> {
                let mut values = self.f64_values(); values[0] = lr;
                self.restore_precise_configuration(&encode($kind,values,self.amsgrad,self.maximize,self.max_grad_norm.is_some()))
            }
            /// Exact CPU scalar settings. Float32 members retain the legacy
            /// arithmetic after validated scalar conversion; Double members use f64.
            pub fn with_f64_options(mut self, lr: f64, beta1: f64, beta2: f64, eps: f64,
                weight_decay: f64, amsgrad: bool, maximize: bool, max_grad_norm: Option<f64>) -> Result<Self> {
                self.restore_precise_configuration(&encode($kind,[lr,beta1,beta2,eps,weight_decay,max_grad_norm.unwrap_or(0.)],
                    amsgrad,maximize,max_grad_norm.is_some()))?;
                Ok(self)
            }
            pub(super) fn f64_values(&self) -> [f64;6] {
                self.precise.unwrap_or([self.lr as f64,self.beta1 as f64,self.beta2 as f64,self.eps as f64,
                    self.weight_decay as f64,self.max_grad_norm.unwrap_or(0.) as f64])
            }
            pub(super) fn precise_configuration(&self) -> Option<Vec<f32>> {
                self.precise.map(|v| encode($kind,v,self.amsgrad,self.maximize,self.max_grad_norm.is_some()))
            }
            pub(super) fn restore_precise_configuration(&mut self, c: &[f32]) -> Result<()> {
                if c.len() != 29 || c[..2] != [$kind,64.] || c[2..5].iter().any(|x| ![0.,1.].contains(x))
                    || c[5..].iter().any(|x| !x.is_finite() || *x < 0. || *x > 65535. || x.fract() != 0.) {
                    return Err(invalid("invalid exact Adam configuration"));
                }
                let mut values = [0.;6];
                for (value,chunks) in values.iter_mut().zip(c[5..].chunks_exact(4)) {
                    *value = f64::from_bits(chunks.iter().enumerate().fold(0u64,|bits,(i,x)| bits | (*x as u64) << (i*16)));
                }
                validate_real_parameters(&self.params)?;
                validate_scalars(values,&self.params)?;
                if (c[4] == 0. && values[5].to_bits() != 0) || self.precise_capturable()
                    || self.params.iter().any(|p| p.tensor().device() != Device::Cpu) {
                    return Err(invalid("exact Adam configuration requires eager CPU parameters and a canonical clipping flag"));
                }
                self.lr = values[0] as f32; self.beta1 = values[1] as f32; self.beta2 = values[2] as f32;
                self.eps = values[3] as f32; self.weight_decay = values[4] as f32;
                self.amsgrad = c[2] == 1.; self.maximize = c[3] == 1.;
                self.max_grad_norm = (c[4] == 1.).then_some(values[5] as f32); self.precise = Some(values);
                Ok(())
            }
            #[doc(hidden)]
            pub fn validate_parameters(&self) -> Result<()> { self.validate_state_parameters(true) }
            fn validate_state_parameters(&self, updating: bool) -> Result<()> {
                validate_real_parameters(&self.params)?;
                if let Some(values) = self.precise {
                    validate_scalars(values,&self.params)?;
                    if self.precise_capturable() || self.params.iter().any(|p| p.tensor().device() != Device::Cpu)
                        || crate::capture::is_recording() {
                        return Err(invalid("exact Adam requires eager CPU execution"));
                    }
                } else if self.params.iter().any(|p| p.tensor().dtype() == DType::F64) {
                    return Err(invalid("float64 Adam parameters require an exact scalar configuration"));
                }
                for (i,p) in self.params.iter().enumerate() {
                    let t = p.tensor();
                    if t.device() == Device::Cpu { crate::checkpoint::validate_destination(&t,&t)?; }
                    for buffer in [&self.m[i],&self.v[i],&self.vmax[i]].into_iter().flatten() {
                        if buffer.dtype() != t.dtype() || buffer.shape() != t.shape() || buffer.device() != t.device() {
                            return Err(invalid("Adam parameter dtype or shape changed after state initialization"));
                        }
                        if buffer.device() == Device::Cpu { crate::checkpoint::validate_destination(buffer,buffer)?; }
                    }
                    if let Some(g) = p.grad() {
                        if g.dtype() != t.dtype() || g.shape() != t.shape() || g.device() != t.device() {
                            return Err(invalid("Adam gradients must match parameter shape, dtype and device"));
                        }
                        if updating && self.t[i] == u32::MAX { return Err(invalid("Adam step counter exhausted")); }
                    }
                }
                Ok(())
            }
            pub(super) fn precise_clip_scale(&self) -> f64 {
                if self.max_grad_norm.is_none() { return 1.; }
                let norm = self.params.iter().filter_map(Param::grad).flat_map(|g|g.to_vec_f64()).map(|x|x*x).sum::<f64>().sqrt();
                let max = self.f64_values()[5];
                if norm > max && norm > 0. { max / norm } else { 1. }
            }
            pub(super) fn update_f64_slot(&mut self, i: usize, gradient: &Tensor, scale: f64) -> Result<()> {
                let parameter = self.params[i].tensor();
                let [lr,beta1,beta2,eps,decay,_] = self.f64_values();
                let step = self.t[i].checked_add(1).ok_or_else(|| invalid("Adam counter exhausted"))?;
                let correction1 = 1. - beta1.powf(step as f64);
                let correction2 = (1. - beta2.powf(step as f64)).sqrt();
                let step_size = lr / correction1;
                let values = parameter.to_vec_f64();
                let old_m = self.m[i].as_ref().map(Tensor::to_vec_f64).unwrap_or_else(||vec![0.;values.len()]);
                let old_v = self.v[i].as_ref().map(Tensor::to_vec_f64).unwrap_or_else(||vec![0.;values.len()]);
                let old_max = if self.amsgrad { self.vmax[i].as_ref().map(Tensor::to_vec_f64).unwrap_or_else(||vec![0.;values.len()]) } else { Vec::new() };
                let mut next_p = Vec::with_capacity(values.len()); let mut next_m = Vec::with_capacity(values.len());
                let mut next_v = Vec::with_capacity(values.len()); let mut next_max = Vec::with_capacity(old_max.len());
                for (j,mut g) in gradient.to_vec_f64().into_iter().enumerate() {
                    if scale != 1. { g *= scale; }
                    if self.maximize { g = -g; }
                    if !$decoupled && decay != 0. { g = decay.mul_add(values[j],g); }
                    // Portable scalar CPU lerp staging. SIMD Torch kernels can
                    // differ in the sign of exact zero at weight == 1.
                    let weight = 1. - beta1; let delta = g - old_m[j];
                    let m = if weight.abs() < 0.5 { weight.mul_add(delta,old_m[j]) }
                        else { (-(1.-weight)).mul_add(delta,g) };
                    let v = ((1.-beta2)*g).mul_add(g,old_v[j]*beta2);
                    let variance = if self.amsgrad {
                        let maximum = if old_max[j].is_nan() { old_max[j] } else if v.is_nan() || v >= old_max[j] { v } else { old_max[j] };
                        next_max.push(maximum); maximum
                    } else { v };
                    let denominator = variance.sqrt() / correction2 + eps;
                    let p = if $decoupled { values[j]*(1.-lr*decay) } else { values[j] };
                    next_p.push(p + ((-step_size*m)/denominator)); next_m.push(m); next_v.push(v);
                }
                let shape = parameter.shape();
                let p = Tensor::from_vec_f64(next_p,shape)?;
                let mut m = Some(Tensor::from_vec_f64(next_m,shape)?); let mut v = Some(Tensor::from_vec_f64(next_v,shape)?);
                let mut maximum = if self.amsgrad { Some(Tensor::from_vec_f64(next_max,shape)?) } else { None };
                commit_buffers(std::slice::from_ref(&self.m[i]),std::slice::from_mut(&mut m))?;
                commit_buffers(std::slice::from_ref(&self.v[i]),std::slice::from_mut(&mut v))?;
                if self.amsgrad { commit_buffers(std::slice::from_ref(&self.vmax[i]),std::slice::from_mut(&mut maximum))?; }
                crate::inplace::raw_copy_("adam_step",&parameter,&p)?;
                self.m[i] = m; self.v[i] = v; self.vmax[i] = maximum; self.t[i] = step; self.params[i].zero_grad();
                Ok(())
            }
            pub(super) fn precise_snapshot(&self) -> Vec<(String,Tensor)> {
                let mut result = Vec::new();
                if $decoupled { result.push(("t".into(),Tensor::from_vec_f64(self.t.iter().map(|&x|x as f64).collect(),&[self.t.len()]).expect("steps"))); }
                for (i,p) in self.params.iter().enumerate() {
                    let t = p.tensor();
                    for (name,slot) in [("m",&self.m[i]),("v",&self.v[i]),("vmax",&self.vmax[i])] {
                        if name == "vmax" && !self.amsgrad { continue; }
                        let value = slot.as_ref().map(Tensor::owned_detach_copy).unwrap_or_else(|| {
                            if t.dtype() == DType::F64 { Tensor::from_vec_f64(vec![0.;t.numel()],t.shape()).expect("parameter") }
                            else { Tensor::zeros(t.shape()) }
                        });
                        result.push((format!("{name}.{i}"),value.reshape(&[value.numel()]).expect("flat")));
                    }
                    if !$decoupled { result.push((format!("t.{i}"),Tensor::from_vec_f64(vec![self.t[i] as f64],&[]).expect("step"))); }
                }
                result
            }
            pub(super) fn restore_precise_staged(&mut self, tensors: &[(String,Tensor)]) -> Result<()> {
                self.validate_state_parameters(false)?;
                let expected = (2 + self.amsgrad as usize + (!$decoupled) as usize)*self.params.len() + $decoupled as usize;
                if tensors.len() != expected { return Err(invalid("wrong Adam state array count")); }
                let counts = if $decoupled {
                    let value = &tensors.iter().find(|(n,_)| n == "t").ok_or_else(||invalid("missing AdamW step"))?.1;
                    if value.shape().is_empty() { vec![steps(value,&[])?[0];self.params.len()] } else { steps(value,&[self.params.len()])? }
                } else {
                    (0..self.params.len()).map(|i| {
                        let value = &tensors.iter().find(|(n,_)| n == &format!("t.{i}")).ok_or_else(||invalid("missing Adam step"))?.1;
                        Ok(steps(value,&[])?[0])
                    }).collect::<Result<Vec<_>>>()?
                };
                for i in 0..self.params.len() {
                    let p = self.params[i].tensor();
                    self.m[i] = Some(state_buffer(tensors,&format!("m.{i}"),&p)?);
                    self.v[i] = Some(state_buffer(tensors,&format!("v.{i}"),&p)?);
                    self.vmax[i] = if self.amsgrad { Some(state_buffer(tensors,&format!("vmax.{i}"),&p)?) } else { None };
                }
                self.t = counts; Ok(())
            }
        }
    }
}
impl Adam { fn precise_capturable(&self) -> bool { false } }
impl AdamW { fn precise_capturable(&self) -> bool { self.capturable_t.is_some() } }
precise_adam!(Adam,2.,false);
precise_adam!(AdamW,3.,true);
