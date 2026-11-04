use super::*;
use crate::DType;

fn invalid(msg: &str) -> Error { Error::Format { op: "sgd", msg: msg.into() } }

impl Sgd {
    pub fn lr_f64(&self) -> f64 { self.f64_values()[0] }
    pub fn set_lr_f64(&mut self, lr: f64) -> Result<()> {
        let mut values = self.f64_values(); values[0] = lr;
        self.restore_configuration(&Self::encode_f64(values,self.nesterov,self.maximize,
            self.max_grad_norm_f64 || self.max_grad_norm.is_some()))
    }
    /// Exact CPU f64 scalar settings. Legacy f32 constructors and kernels keep
    /// their original arithmetic; mixed groups round only their f32 slots.
    pub fn with_f64_options(mut self, lr: f64, momentum: f64, weight_decay: f64,
        dampening: f64, nesterov: bool, maximize: bool, max_grad_norm: Option<f64>) -> Result<Self> {
        let config = Self::encode_f64([lr, momentum, weight_decay, dampening, max_grad_norm.unwrap_or(0.)],
            nesterov, maximize, max_grad_norm.is_some());
        self.restore_configuration(&config)?;
        Ok(self)
    }

    fn encode_f64(values: [f64; 5], nesterov: bool, maximize: bool, clip: bool) -> Vec<f32> {
        let mut config = vec![1., 64., nesterov as u8 as f32, maximize as u8 as f32, clip as u8 as f32];
        for value in values { let bits = value.to_bits(); for shift in [0,16,32,48] { config.push(((bits >> shift) & 65535) as f32); } }
        config
    }

    pub(super) fn precise_configuration(&self) -> Option<Vec<f32>> {
        self.precise.map(|values| Self::encode_f64(values, self.nesterov, self.maximize, self.max_grad_norm_f64))
    }

    pub(super) fn restore_precise_configuration(&mut self, config: &[f32]) -> Result<()> {
        if config.len() != 25 || config[..2] != [1.,64.]
            || config[2..5].iter().any(|x| ![0.,1.].contains(x))
            || config[5..].iter().any(|x| !x.is_finite() || *x < 0. || *x > 65535. || x.fract() != 0.) {
            return Err(invalid("invalid exact SGD configuration"));
        }
        let mut values = [0.;5];
        for (value, chunks) in values.iter_mut().zip(config[5..].chunks_exact(4)) {
            let bits = chunks.iter().enumerate().fold(0u64, |bits,(i,part)| bits | (*part as u64) << (16*i));
            *value = f64::from_bits(bits);
        }
        if values.iter().any(|x| !x.is_finite() || *x < 0.)
            || (config[4] == 0. && values[4].to_bits() != 0)
            || (config[2] == 1. && (values[1] <= 0. || values[3] != 0.)) {
            return Err(invalid("invalid exact SGD scalars or flags"));
        }
        if self.params.iter().any(|p| p.tensor().dtype() == DType::F32)
            && values.iter().enumerate().any(|(i,&x)| !(x as f32).is_finite() || (matches!(i,1|3) && x != 0. && x as f32 == 0.)) {
            return Err(invalid("SGD scalars for float32 parameters must fit float32"));
        }
        self.lr = values[0] as f32; self.momentum = values[1] as f32;
        self.weight_decay = values[2] as f32; self.dampening = values[3] as f32;
        self.nesterov = config[2] == 1.; self.maximize = config[3] == 1.;
        self.max_grad_norm = (config[4] == 1.).then_some(values[4] as f32);
        self.max_grad_norm_f64 = config[4] == 1.; self.precise = Some(values);
        Ok(())
    }

    pub(super) fn f64_values(&self) -> [f64;5] {
        self.precise.unwrap_or([self.lr as f64,self.momentum as f64,self.weight_decay as f64,
            self.dampening as f64,self.max_grad_norm.unwrap_or(0.) as f64])
    }

    pub(super) fn has_initialization_flags(&self) -> bool { self.precise.is_some() || self.dampening != 0. }

    #[doc(hidden)]
    pub fn validate_parameters(&self) -> Result<()> {
        if let Some(values) = self.precise {
            if self.params.iter().any(|p| p.tensor().dtype() == DType::F32)
                && values.iter().enumerate().any(|(i,&x)| !(x as f32).is_finite() || (matches!(i,1|3) && x != 0. && x as f32 == 0.)) {
                return Err(invalid("SGD scalars for float32 parameters must fit float32"));
            }
        }
        for (p,buffer) in self.params.iter().zip(&self.velocity) {
            let t = p.tensor();
            if t.dtype() != DType::F32 && !(t.dtype() == DType::F64 && t.device() == Device::Cpu) {
                return Err(invalid("SGD requires float32 parameters or CPU float64 parameters"));
            }
            if t.dtype() == DType::F64 { crate::checkpoint::validate_destination(&t,&t)?; }
            if let Some(buffer) = buffer {
                if buffer.dtype() != t.dtype() || buffer.shape() != t.shape() || buffer.device() != t.device() {
                    return Err(invalid("SGD parameter dtype or shape changed after state initialization"));
                }
                if t.device() == Device::Cpu { crate::checkpoint::validate_destination(buffer,buffer)?; }
            }
            if let Some(g) = p.grad() {
                if g.dtype() != t.dtype() || g.shape() != t.shape() || g.device() != t.device() {
                    return Err(invalid("SGD gradients must match parameter dtype, shape and device"));
                }
            }
        }
        if self.params.iter().any(|p| p.tensor().dtype() == DType::F64) && crate::capture::is_recording() {
            return Err(Error::Unsupported { op: "sgd", msg: "float64 SGD requires eager CPU execution".into() });
        }
        Ok(())
    }

    pub(super) fn f64_clip_scale(&self) -> f64 {
        if !self.max_grad_norm_f64 && self.max_grad_norm.is_none() { return 1.; }
        let norm = self.params.iter().filter_map(Param::grad).flat_map(|g| g.to_vec_f64()).map(|x|x*x).sum::<f64>().sqrt();
        (self.f64_values()[4] / (norm + 1e-6)).min(1.)
    }

    pub(super) fn update_f64_slot(&mut self, i: usize, gradient: &Tensor, scale: f64) -> Result<()> {
        let parameter = self.params[i].tensor();
        let [lr,momentum,decay,dampening,_] = self.f64_values();
        let values = parameter.to_vec_f64();
        let mut grads = gradient.to_vec_f64();
        for (g,&p) in grads.iter_mut().zip(&values) {
            if scale != 1. { *g *= scale; }
            if self.maximize { *g = -*g; }
            if decay != 0. { *g = decay.mul_add(p,*g); }
        }
        let mut pending = None;
        if momentum != 0. {
            let next = if self.initialized[i] {
                let old = self.velocity[i].as_ref().ok_or_else(|| invalid("missing initialized SGD momentum"))?.to_vec_f64();
                old.into_iter().zip(&grads).map(|(old,&g)| (1.-dampening).mul_add(g,old*momentum)).collect::<Vec<_>>()
            } else { grads.clone() };
            for (g,&v) in grads.iter_mut().zip(&next) { *g = if self.nesterov { momentum.mul_add(v,*g) } else { v }; }
            pending = Some(Tensor::from_vec_f64(next,parameter.shape())?);
        }
        let next = Tensor::from_vec_f64(values.into_iter().zip(grads).map(|(p,g)| (-lr).mul_add(g,p)).collect(),parameter.shape())?;
        if pending.is_some() {
            commit_buffers(std::slice::from_ref(&self.velocity[i]),std::slice::from_mut(&mut pending))?;
        }
        crate::inplace::raw_copy_("sgd_step",&parameter,&next)?;
        if momentum != 0. { self.velocity[i] = pending; self.initialized[i] = true; }
        self.params[i].zero_grad();
        Ok(())
    }
}
