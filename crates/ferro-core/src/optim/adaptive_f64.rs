//! Exact CPU scalar/configuration and Float64 updates for Adaptive tags 5/6.
use super::*;

pub(super) fn validate_parameter_types(params: &[Param], kind: u8) -> Result<()> {
    if !matches!(kind,5|6) { return validate_cpu_f32_parameters(params); }
    for p in params {
        let value=p.tensor();
        if !matches!(value.dtype(),DType::F32|DType::F64) || value.device()!=Device::Cpu {
            return Err(invalid("RMSprop/Adagrad require CPU float32 or float64 parameters"));
        }
        crate::checkpoint::validate_destination(&value,&value)?;
    }
    Ok(())
}

fn encode(kind:u8, values:[f64;5], centered:bool, maximize:bool)->Vec<f32> {
    let mut result=vec![kind as f32,64.,centered as u8 as f32,maximize as u8 as f32,0.];
    for value in values { for shift in [0,16,32,48] {result.push(((value.to_bits()>>shift)&65535) as f32);} }
    result
}

pub(super) fn decode(config:&[f32], params:&[Param])->Result<(Vec<f32>,Option<[f64;5]>)> {
    if config.len()!=25 {
        validate_config(config)?;
        let kind=config[0] as u8;validate_parameter_types(params,kind)?;
        let precise=if params.iter().any(|p|p.tensor().dtype()==DType::F64) {
            Some([config[1] as f64,config[2] as f64,config[3] as f64,config[4] as f64,config[5] as f64])
        } else {None};
        return Ok((config.to_vec(),precise));
    }
    if ![5.,6.].contains(&config[0]) || config[1]!=64. || config[2..4].iter().any(|x|![0.,1.].contains(x))
        || config[4].to_bits()!=0 || (config[0]==6. && config[2]!=0.)
        || config[5..].iter().any(|x|!x.is_finite()||*x<0.||*x>65535.||x.fract()!=0.) {
        return Err(invalid("invalid exact adaptive configuration"));
    }
    let kind=config[0] as u8;validate_parameter_types(params,kind)?;
    let mut values=[0.;5];
    for (value,chunks) in values.iter_mut().zip(config[5..].chunks_exact(4)) {
        *value=f64::from_bits(chunks.iter().enumerate().fold(0u64,|bits,(i,x)|bits|(*x as u64)<<(i*16)));
    }
    if values.iter().any(|x|!x.is_finite()||*x<0.) {return Err(invalid("exact adaptive scalars must be finite and nonnegative"));}
    if params.iter().any(|p|p.tensor().dtype()==DType::F32)
        && (values.iter().any(|&x|!(x as f32).is_finite())
            || (kind==5 && (values[1]>1. || (values[4]!=0.&&values[4] as f32==0.)))) {
        return Err(invalid("adaptive scalars for float32 members exceed the existing float32 range or option limits"));
    }
    let mut legacy=vec![kind as f32];legacy.extend(values.iter().map(|&x|x as f32));
    if kind==5 {legacy.push(config[2]);}
    legacy.push(config[3]);
    Ok((legacy,Some(values)))
}

impl Adaptive {
    /// Exact CPU RMSprop or Adagrad options using the existing Adaptive state.
    /// Tag 5 values: lr, alpha, eps, weight_decay, momentum. Tag 6 values:
    /// lr, lr_decay, weight_decay, initial_accumulator_value, eps.
    pub fn new_f64(params:Vec<Param>, kind:u8, values:[f64;5], centered:bool, maximize:bool)->Result<Self> {
        Self::new(params,&encode(kind,values,centered,maximize))
    }
    pub fn lr_f64(&self)->f64 {self.precise.map(|v|v[0]).unwrap_or(self.lr() as f64)}
    pub fn set_lr_f64(&mut self, lr:f64)->Result<()> {
        if !matches!(self.kind(),5|6) {return Err(invalid("exact scalar updates are supported by RMSprop and Adagrad"));}
        let mut values=self.f64_values();values[0]=lr;
        self.restore_configuration(&encode(self.kind(),values,self.kind()==5&&self.config[6]==1.,*self.config.last().unwrap()==1.))
    }
    fn f64_values(&self)->[f64;5] {
        self.precise.unwrap_or([self.config[1] as f64,self.config[2] as f64,self.config[3] as f64,self.config[4] as f64,self.config[5] as f64])
    }
    pub(super) fn precise_configuration(&self)->Option<Vec<f32>> {
        self.precise.map(|v|encode(self.kind(),v,self.kind()==5&&self.config[6]==1.,*self.config.last().unwrap()==1.))
    }
    /// Metadata/scalar validation only: config sync must not allocate Adagrad's
    /// eagerly initialized accumulator for a second temporary optimizer.
    #[doc(hidden)]
    pub fn validate_configuration(&self,config:&[f32])->Result<()> {
        let (decoded,_)=decode(config,&self.params)?;
        if decoded[0]!=self.config[0] {return Err(invalid("optimizer family mismatch"));}
        Ok(())
    }
    #[doc(hidden)]
    pub fn validate_parameters(&self)->Result<()> {
        validate_parameter_types(&self.params,self.kind())?;
        // Revalidate exact settings against current descriptors, including a
        // changed Float32 member in a formerly all-Double group.
        if let Some(config)=self.precise_configuration() {decode(&config,&self.params)?;}
        for (i,p) in self.params.iter().enumerate() {
            let parameter=p.tensor();validate_cpu_buffers(&self.buffers[i])?;
            for (field,value) in self.buffers[i].iter().enumerate() {
                if let Some(value)=value {
                    let shape=if self.is_scalar(field){vec![]}else{parameter.shape().to_vec()};
                    if value.dtype()!=parameter.dtype()||value.shape()!=shape {return Err(invalid("adaptive parameter dtype or shape changed after state initialization"));}
                }
            }
            if let Some(g)=p.grad() {
                if g.device()!=Device::Cpu||g.dtype()!=parameter.dtype()||g.shape()!=parameter.shape() {return Err(invalid("adaptive gradient must match parameter dtype, device and shape"));}
                if self.steps[i]>=MAX_STEP {return Err(invalid("step exceeds exact f32 checkpoint counter range"));}
            }
        }
        Ok(())
    }
    pub(super) fn initial_buffer(&self,parameter:&Tensor,field:usize)->Result<Tensor> {
        let shape=if self.is_scalar(field){&[][..]}else{parameter.shape()};
        if parameter.dtype()==DType::F64 {
            let initial=if self.kind()==6 {self.f64_values()[3]}else{0.};
            Tensor::from_vec_f64(vec![initial;parameter.numel()],shape)
        } else {Ok(Tensor::full(shape,self.initial(field)))}
    }
    pub(super) fn stage_f64(&self,i:usize,gradient:&Tensor,shadow:&StagedParameterValues)->Result<(Tensor,Vec<Option<Tensor>>)> {
        let parameter=self.params[i].tensor();let mut values=shadow.read_f64(&parameter);let mut grads=gradient.to_vec_f64();
        let c=self.f64_values();let decay=if self.kind()==5{c[3]}else{c[2]};let maximize=*self.config.last().unwrap()==1.;
        for (g,&p) in grads.iter_mut().zip(&values) {if maximize{*g=-*g;}if decay!=0.{*g=decay.mul_add(p,*g);}}
        let mut state:Vec<Vec<f64>>=self.buffers[i].iter().map(|t|t.as_ref().map(Tensor::to_vec_f64).unwrap_or_else(||vec![if self.kind()==6{c[3]}else{0.};values.len()])).collect();
        match self.kind() {
            5=>{
                let [lr,alpha,eps,_,momentum]=c;let centered=self.config[6]==1.;
                for (j,(p,&g)) in values.iter_mut().zip(&grads).enumerate() {
                    state[0][j]=((1.-alpha)*g).mul_add(g,alpha*state[0][j]);
                    let variance=if centered {
                        let weight=1.-alpha;let delta=g-state[2][j];
                        state[2][j]=if weight.abs()<0.5{weight.mul_add(delta,state[2][j])}else{(-(1.-weight)).mul_add(delta,g)};
                        (-state[2][j]).mul_add(state[2][j],state[0][j])
                    } else {state[0][j]};
                    let denominator=variance.sqrt()+eps;
                    if momentum>0. {state[1][j]=state[1][j]*momentum+g/denominator;*p=(-lr).mul_add(state[1][j],*p);}
                    else {*p+=(-lr*g)/denominator;}
                }
            }
            6=>{
                let [lr,lr_decay,_,_,eps]=c;let rate=lr/(1.+self.steps[i] as f64*lr_decay);
                for (j,(p,&g)) in values.iter_mut().zip(&grads).enumerate() {
                    state[0][j]=g.mul_add(g,state[0][j]);*p+=(-rate*g)/(state[0][j].sqrt()+eps);
                }
            }
            _=>return Err(invalid("float64 adaptive update is unsupported for this family")),
        }
        let next=Tensor::from_vec_f64(values,parameter.shape())?;
        let next_state=state.into_iter().map(|v|Tensor::from_vec_f64(v,parameter.shape()).map(Some)).collect::<Result<_>>()?;
        Ok((next,next_state))
    }
}
