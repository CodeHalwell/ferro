//! CPU adaptive optimizers. RMSprop/Adagrad also support Float64 parameters.
//! Updates preserve parameter and moment storage,
//! consume active gradients, and do not participate in graph capture.
use super::{commit_buffers, validate_buffer_copies, validate_cpu_buffers, validate_cpu_f32_parameters, OptimizerState, StagedParameterValues};
use crate::{Device, Error, Param, Result, Tensor};
use crate::dtype::DType;

#[path = "adaptive_f64.rs"]
mod typed;

const MAX_STEP: u32 = 1 << 24;

/// Adaptive optimizer families with CPU Float32 state and typed RMSprop/Adagrad state.
/// Configuration tags 5 through 12 identify the update and serialized schema.
/// Clones share parameter slots but own independent state buffers.
pub struct Adaptive {
    params: Vec<Param>,
    config: Vec<f32>,
    precise: Option<[f64; 5]>,
    steps: Vec<u32>,
    initialized: Vec<bool>,
    buffers: Vec<Vec<Option<Tensor>>>,
}

impl Clone for Adaptive {
    fn clone(&self) -> Self {
        let mut cloned = self.clone_for_restore();
        cloned.buffers = self.buffers.iter().map(|slot| slot.iter().map(|t| t.as_ref().map(Tensor::owned_detach_copy)).collect()).collect();
        cloned
    }
}

fn invalid(message: &str) -> Error { Error::Format { op: "adaptive_optimizer", msg: message.into() } }

fn validate_config(c: &[f32]) -> Result<()> {
    let width = match c.first().copied() { Some(5.0) | Some(11.0) => 8, Some(6.0) | Some(7.0) | Some(9.0) | Some(10.0) => 7,
        Some(8.0) => 6, Some(12.0) => 9, _ => return Err(invalid("unknown optimizer configuration")) };
    if c.len() != width || c.iter().any(|x| !x.is_finite() || *x < 0.0) || ![0.0, 1.0].contains(c.last().unwrap()) {
        return Err(invalid("invalid optimizer configuration"));
    }
    if (c[0] == 5.0 && (c[2] > 1.0 || ![0.0, 1.0].contains(&c[6])))
        || ([7.0, 11.0, 12.0].contains(&c[0]) && (c[2] >= 1.0 || c[3] >= 1.0)) || (c[0] == 8.0 && c[2] > 1.0)
        || (c[0] == 10.0 && (c[2] <= 0.0 || c[2] >= 1.0 || c[3] <= 1.0 || c[5] < c[4]))
        || (c[0] == 11.0 && ![0.0, 1.0].contains(&c[6])) || (c[0] == 12.0 && ![0.0, 1.0].contains(&c[7])) {
        return Err(invalid("invalid optimizer decay or flag"));
    }
    Ok(())
}

impl Adaptive {
    /// Configurations are [tag, lr, ...]: RMSprop [5, lr, alpha, eps, decay,
    /// momentum, centered, maximize], Adagrad [6, lr, lr_decay, decay, initial,
    /// eps, maximize], Adamax [7, lr, beta1, beta2, eps, decay, maximize],
    /// Adadelta [8, lr, rho, eps, decay, maximize], ASGD [9, lr, lambd, alpha,
    /// t0, decay, maximize]. Flags are exactly 0 or 1; decay is coupled L2.
    pub fn new(params: Vec<Param>, config: &[f32]) -> Result<Self> {
        let mut seen = std::collections::HashSet::new();
        let params: Vec<_> = params.into_iter().filter(|p| seen.insert(p.identity())).collect();
        let (config, precise) = typed::decode(config, &params)?;
        let mut this = Self { steps: vec![0; params.len()], initialized: vec![false; params.len()], buffers: vec![], params, config, precise };
        this.buffers = this.params.iter().map(|p| this.names().iter().map(|_| {
            if this.kind() == 6 { Some(this.initial_buffer(&p.tensor(), 0).expect("validated accumulator")) } else { None }
        }).collect()).collect();
        Ok(this)
    }

    fn clone_for_restore(&self) -> Self {
        Self { params: self.params.clone(), config: self.config.clone(), precise: self.precise, steps: self.steps.clone(), initialized: self.initialized.clone(), buffers: self.buffers.clone() }
    }
    fn kind(&self) -> u8 { self.config[0] as u8 }
    fn names(&self) -> &'static [&'static str] {
        match self.kind() {
            5 => &["square_avg", "momentum_buffer", "grad_avg"],
            6 => &["sum"], 7 => &["exp_avg", "exp_inf"], 8 => &["square_avg", "acc_delta"],
            9 => &["ax", "eta", "mu"], 10 => &["prev", "step_size"],
            11 => &["exp_avg", "exp_avg_sq"], 12 => &["exp_avg", "exp_avg_sq", "mu_product"], _ => unreachable!(),
        }
    }
    fn is_scalar(&self, field: usize) -> bool { (self.kind() == 9 && field > 0) || (self.kind() == 12 && field == 2) }
    fn has_initialized_state(&self) -> bool { matches!(self.kind(), 9 | 10) }
    fn initial(&self, field: usize) -> f32 {
        match (self.kind(), field) { (6, 0) => self.config[4], (9, 1) | (10, 1) => self.config[1], (9, 2) | (12, 2) => 1.0, _ => 0.0 }
    }
    pub fn lr(&self) -> f32 { self.config[1] }
    pub fn set_lr(&mut self, lr: f32) -> Result<()> {
        if self.precise.is_some() { return self.set_lr_f64(lr as f64); }
        let mut config = self.config.clone(); config[1] = lr;
        self.restore_configuration(&config)
    }
    pub fn zero_grad(&self) { for p in &self.params { p.zero_grad(); } }

    /// All metadata and counter checks precede mutation. CPU state and parameter
    /// copies are then infallible; no backend/device transfer is attempted.
    pub fn step(&mut self) -> Result<()> {
        if crate::capture::is_recording() {
            return Err(Error::Unsupported { op: "adaptive_optimizer", msg: "optimizer steps do not support graph capture".into() });
        }
        super::validate_gradient_storage(&self.params)?;
        self.validate_parameters()?;
        for (i, p) in self.params.iter().enumerate() {
            validate_cpu_buffers(&self.buffers[i])?;
            for (j, buffer) in self.buffers[i].iter().enumerate() {
                if let Some(t) = buffer {
                    let shape = if self.is_scalar(j) { vec![] } else { p.tensor().shape().to_vec() };
                    if t.shape() != shape { return Err(invalid("parameter shape changed after optimizer state initialization")); }
                }
            }
            if let Some(g) = p.grad() {
                if g.device() != Device::Cpu || g.dtype() != p.tensor().dtype() || g.shape() != p.tensor().shape() {
                    return Err(invalid("gradients must match CPU parameter dtype and shape"));
                }
                if self.steps[i] >= MAX_STEP { return Err(invalid("step exceeds exact f32 checkpoint counter range")); }
            }
        }
        let mut pending = Vec::new();
        let mut staged_values = StagedParameterValues::default();
        for (i, p) in self.params.iter().enumerate() {
            let Some(gradient) = p.grad() else { continue };
            let parameter = p.tensor();
            if parameter.dtype() == DType::F64 {
                let (next, next_state) = self.stage_f64(i, &gradient, &staged_values)?;
                staged_values.stage(&parameter, &next);
                pending.push((i, parameter, next, next_state));
                continue;
            }
            let mut values = staged_values.read(&parameter, || parameter.to_vec());
            let mut grads = gradient.to_vec();
            let step = self.steps[i] + 1;
            let c = &self.config;
            let decay = match self.kind() { 5 | 8 => c[4], 6 => c[3], 7 | 9 | 11 | 12 => c[5], 10 => 0.0, _ => unreachable!() };
            let decoupled = (self.kind() == 11 && c[6] == 1.0) || (self.kind() == 12 && c[7] == 1.0);
            for (g, value) in grads.iter_mut().zip(&values) {
                if *c.last().unwrap() == 1.0 { *g = -*g; }
                if decay != 0.0 && !decoupled { *g += decay * *value; }
            }
            if decoupled && decay != 0.0 {
                let scale = (1.0 - c[1] as f64 * decay as f64) as f32;
                for value in &mut values { *value *= scale; }
            }
            let mut state: Vec<Vec<f32>> = self.buffers[i].iter().enumerate().map(|(field, t)| {
                t.as_ref().map(Tensor::to_vec).unwrap_or_else(|| vec![self.initial(field); if self.is_scalar(field) { 1 } else { values.len() }])
            }).collect();
            match self.kind() {
                5 => {
                    let (lr, alpha, eps, momentum, centered) = (c[1], c[2], c[3], c[5], c[6] == 1.0);
                    for (j, (p, g)) in values.iter_mut().zip(&grads).enumerate() {
                        state[0][j] = alpha * state[0][j] + (1.0 - alpha) * g * g;
                        let variance = if centered {
                            state[2][j] += (1.0 - alpha) * (g - state[2][j]);
                            state[0][j] - state[2][j] * state[2][j]
                        } else { state[0][j] };
                        let denominator = variance.sqrt() + eps;
                        let update = if momentum > 0.0 {
                            state[1][j] = momentum * state[1][j] + g / denominator;
                            state[1][j]
                        } else { g / denominator };
                        *p -= lr * update;
                    }
                }
                6 => {
                    let rate = (c[1] as f64 / (1.0 + (step - 1) as f64 * c[2] as f64)) as f32;
                    for (j, (p, g)) in values.iter_mut().zip(&grads).enumerate() {
                        state[0][j] += g * g;
                        *p -= rate * g / (state[0][j].sqrt() + c[5]);
                    }
                }
                7 => {
                    let rate = (c[1] as f64 / (1.0 - (c[2] as f64).powf(step as f64))) as f32;
                    for (j, (p, g)) in values.iter_mut().zip(&grads).enumerate() {
                        state[0][j] += (1.0 - c[2]) * (g - state[0][j]);
                        let old = c[3] * state[1][j]; let new = g.abs() + c[4];
                        state[1][j] = if old.is_nan() || new.is_nan() { f32::NAN } else { old.max(new) };
                        *p -= rate * state[0][j] / state[1][j];
                    }
                }
                8 => {
                    for (j, (p, g)) in values.iter_mut().zip(&grads).enumerate() {
                        state[0][j] = c[2] * state[0][j] + (1.0 - c[2]) * g * g;
                        let delta = ((state[1][j] + c[3]).sqrt() / (state[0][j] + c[3]).sqrt()) * g;
                        state[1][j] = c[2] * state[1][j] + (1.0 - c[2]) * delta * delta;
                        *p -= c[1] * delta;
                    }
                }
                9 => {
                    if !self.initialized[i] { state[1][0] = c[1]; state[2][0] = 1.0; }
                    let (eta, mu) = (state[1][0], state[2][0]);
                    let shrink = (1.0 - c[2] as f64 * eta as f64) as f32;
                    for (j, (p, g)) in values.iter_mut().zip(&grads).enumerate() {
                        *p *= shrink;
                        *p -= eta * g;
                        state[0][j] = if mu == 1.0 { *p } else { state[0][j] + (*p - state[0][j]) * mu };
                    }
                    state[1][0] = (c[1] as f64 / (1.0 + c[2] as f64 * c[1] as f64 * step as f64).powf(c[3] as f64)) as f32;
                    state[2][0] = (1.0 / (step as f64 - c[4] as f64).max(1.0)) as f32;
                }
                10 => {
                    if !self.initialized[i] { state[1].fill(c[1]); }
                    for (j, (p, g)) in values.iter_mut().zip(&grads).enumerate() {
                        let product = *g * state[0][j];
                        let factor = if product > 0.0 { c[3] } else if product < 0.0 { c[2] } else { 1.0 };
                        state[1][j] = (state[1][j] * factor).clamp(c[4], c[5]);
                        let gradient = if product < 0.0 { 0.0 } else { *g };
                        let direction = if gradient > 0.0 { 1.0 } else if gradient < 0.0 { -1.0 } else { 0.0 };
                        *p -= direction * state[1][j];
                        state[0][j] = gradient;
                    }
                }
                11 => {
                    let beta1 = c[2] as f64; let beta2 = c[3] as f64;
                    let correction1 = (1.0 - beta1.powf(step as f64)) as f32;
                    let correction2 = 1.0 - beta2.powf(step as f64);
                    let rho_inf = 2.0 / (1.0 - beta2) - 1.0;
                    let rho = rho_inf - 2.0 * step as f64 * beta2.powf(step as f64) / correction2;
                    let rectification = if rho > 5.0 { (((rho - 4.0) * (rho - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho)).sqrt() as f32 } else { 1.0 };
                    for (j, (p, g)) in values.iter_mut().zip(&grads).enumerate() {
                        state[0][j] += (1.0 - c[2]) * (g - state[0][j]);
                        state[1][j] = c[3] * state[1][j] + (1.0 - c[3]) * g * g;
                        let mut update = (state[0][j] / correction1) * c[1];
                        if rho > 5.0 { update = (update * (correction2.sqrt() as f32 / (state[1][j].sqrt() + c[4]))) * rectification; }
                        *p -= update;
                    }
                }
                12 => {
                    let beta1 = c[2] as f64; let decay = c[6] as f64;
                    let mu = beta1 * (1.0 - 0.5 * 0.96f64.powf(step as f64 * decay));
                    let mu_next = beta1 * (1.0 - 0.5 * 0.96f64.powf((step + 1) as f64 * decay));
                    state[2][0] *= mu as f32;
                    let correction2 = (1.0 - (c[3] as f64).powf(step as f64)) as f32;
                    let grad_rate = (-c[1] as f64 * (1.0 - mu) / (1.0 - state[2][0] as f64)) as f32;
                    let mean_rate = (-c[1] as f64 * mu_next / (1.0 - state[2][0] as f64 * mu_next)) as f32;
                    for (j, (p, g)) in values.iter_mut().zip(&grads).enumerate() {
                        state[0][j] += (1.0 - c[2]) * (g - state[0][j]);
                        state[1][j] = c[3] * state[1][j] + (1.0 - c[3]) * g * g;
                        let denominator = (state[1][j] / correction2).sqrt() + c[4];
                        *p += grad_rate * g / denominator;
                        *p += mean_rate * state[0][j] / denominator;
                    }
                }
                _ => unreachable!(),
            }
            let next = Tensor::from_vec(values, parameter.shape())?;
            staged_values.stage(&parameter, &next);
            let next_state: Vec<_> = state.into_iter().enumerate().map(|(field, data)| {
                let shape = if self.is_scalar(field) { &[][..] } else { parameter.shape() };
                Tensor::from_vec(data, shape).map(Some)
            }).collect::<Result<_>>()?;
            pending.push((i, parameter, next, next_state));
        }
        for (i, parameter, next, mut next_state) in pending {
            commit_buffers(&self.buffers[i], &mut next_state).expect("validated CPU optimizer state");
            crate::inplace::raw_copy_("adaptive_optimizer", &parameter, &next).expect("validated CPU parameter");
            self.buffers[i] = next_state;
            self.steps[i] += 1;
            self.initialized[i] = true;
            self.params[i].zero_grad();
        }
        Ok(())
    }

    fn restore_staged(&mut self, state: &[(String, Tensor)]) -> Result<()> {
        let names = self.names();
        if state.len() != self.params.len() * (names.len() + 1 + usize::from(self.has_initialized_state())) { return Err(invalid("optimizer state key count mismatch")); }
        let mut unique = std::collections::HashSet::new();
        if state.iter().any(|(name, _)| !unique.insert(name)) { return Err(invalid("duplicate optimizer state key")); }
        for i in 0..self.params.len() {
            let find = |name: &str| state.iter().find(|(key, _)| key == &format!("{name}.{i}")).map(|(_, t)| t).ok_or_else(|| invalid("missing optimizer state key"));
            let counter = find("t")?;
            if counter.dtype() != DType::F32 || counter.device() != Device::Cpu || counter.shape() != [] { return Err(invalid("step must be scalar CPU f32")); }
            let t = counter.item();
            if !t.is_finite() || t < 0.0 || t.fract() != 0.0 || t > MAX_STEP as f32 { return Err(invalid("step exceeds exact f32 checkpoint counter range")); }
            self.steps[i] = t as u32;
            if self.has_initialized_state() {
                let initialized = find("initialized")?;
                if initialized.dtype() != DType::F32 || initialized.device() != Device::Cpu || initialized.shape() != []
                    || ![0.0, 1.0].contains(&initialized.item()) || (t > 0.0 && initialized.item() != 1.0) {
                    return Err(invalid("invalid optimizer initialized flag"));
                }
                self.initialized[i] = initialized.item() == 1.0;
            } else { self.initialized[i] = t > 0.0; }
            for (field, name) in names.iter().enumerate() {
                let tensor = find(name)?;
                let scalar = self.is_scalar(field);
                let shape = if scalar { vec![] } else { vec![self.params[i].tensor().numel()] };
                if tensor.dtype() != self.params[i].tensor().dtype() || tensor.device() != Device::Cpu || tensor.shape() != shape { return Err(invalid("optimizer buffer shape/dtype/device mismatch")); }
                if scalar && (!tensor.item().is_finite() || tensor.item() < 0.0 || (["mu", "mu_product"].contains(name) && tensor.item() > 1.0)) {
                    return Err(invalid("invalid optimizer scalar state"));
                }
                let shape = if scalar { vec![] } else { self.params[i].tensor().shape().to_vec() };
                self.buffers[i][field] = Some(tensor.owned_detach_copy().reshape(&shape)?);
            }
        }
        Ok(())
    }
}

impl OptimizerState for Adaptive {
    fn state_parameters(&self) -> Option<Vec<Param>> { Some(self.params.clone()) }
    fn state_buffers(&self) -> Option<Vec<Tensor>> { Some(self.buffers.iter().flatten().flatten().cloned().collect()) }
    fn configuration(&self) -> Vec<f32> { self.precise_configuration().unwrap_or_else(||self.config.clone()) }
    fn restore_configuration(&mut self, config: &[f32]) -> Result<()> {
        let (config, precise) = typed::decode(config, &self.params)?;
        if config[0] != self.config[0] { return Err(invalid("optimizer family mismatch")); }
        self.config = config; self.precise = precise;
        Ok(())
    }
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        let mut result = Vec::new();
        for i in 0..self.params.len() {
            result.push((format!("t.{i}"), Tensor::scalar(self.steps[i] as f32)));
            if self.has_initialized_state() { result.push((format!("initialized.{i}"), Tensor::scalar(self.initialized[i] as u8 as f32))); }
            for (field, name) in self.names().iter().enumerate() {
                let shape = if self.is_scalar(field) { vec![] } else { vec![self.params[i].tensor().numel()] };
                let t = self.buffers[i][field].as_ref().map(Tensor::owned_detach_copy).unwrap_or_else(|| self.initial_buffer(&self.params[i].tensor(),field).expect("validated parameter"));
                result.push((format!("{name}.{i}"), t.reshape(&shape).expect("optimizer snapshot shape")));
            }
        }
        result
    }
    fn prepare_cpu_restore<'a>(&'a mut self, config: &[f32], state: &[(String, Tensor)]) -> Result<Box<dyn FnOnce() + 'a>> {
        typed::validate_parameter_types(&self.params, self.kind())?;
        for buffers in &self.buffers { validate_cpu_buffers(buffers)?; }
        let mut staged = self.clone_for_restore();
        staged.restore_configuration(config)?;
        staged.restore_staged(state)?;
        for (old, new) in self.buffers.iter().zip(&staged.buffers) { validate_buffer_copies(old, new)?; }
        Ok(Box::new(move || {
            for (old, new) in self.buffers.iter().zip(&mut staged.buffers) { commit_buffers(old, new).expect("validated CPU optimizer state"); }
            *self = staged;
        }))
    }
    fn restore(&mut self, state: &[(String, Tensor)]) -> Result<()> {
        let config = self.configuration();
        self.prepare_cpu_restore(&config, state)?();
        Ok(())
    }
}
