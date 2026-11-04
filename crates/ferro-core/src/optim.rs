//! Optimizers, LR schedulers, and gradient clipping,
//! operating on parameter tensors and their `.grad()`.
//!
//! Default SGD, Adam and AdamW steps mutate IN PLACE through the fused no-grad seams in `inplace`
//! (`raw_sgd_step_`/`raw_adamw_step_`/`raw_axpy_`): a parameter and its
//! optimizer state (momentum buffers, Adam moments - tensors on the
//! parameter's device) keep their storage addresses across every step, the
//! step allocates nothing for them, and scalars (lr, betas, per-step bias
//! corrections) ride into the kernels as plain f32 arguments - a steady-state
//! device step is one fused kernel launch per parameter with zero host
//! traffic in either direction. The update math is elementwise-identical to
//! the unfused formulas, so results are bitwise unchanged. Mutating a
//! grad-requiring leaf is the engine's `torch.no_grad()` step equivalent:
//! nothing is recorded, the storage version is bumped, and a stale graph that
//! saved the parameter fails loudly on backward instead of silently reusing
//! old values.
//!
//! A parameter the fused seams reject (a strided/aliased leaf, a backend
//! without in-place kernels) falls back to the original allocating update -
//! same numbers, fresh storage, visible through `_storage_ptr` staying
//! unstable. Timestep counters stay host-side since they feed powi, not
//! elementwise math.

// Additional adaptive families use CPU f32 updates. SGD and Adam families
// additionally support eager CPU f64 through their typed update helpers.
mod sgd_f64;
mod adam_f64;
mod adaptive;
pub use adaptive::Adaptive;
mod lbfgs;
pub use lbfgs::Lbfgs;
mod adafactor;
pub use adafactor::Adafactor;
mod muon;
pub use muon::Muon;

use crate::device::Device;
use crate::dispatch::{AdamWStep, BinaryKind, UnaryKind};
use crate::error::Error;
use crate::inplace::{
    raw_adamw_step_, raw_adamw_step_capturable_, raw_axpy_, raw_scalar_increment_, raw_sgd_step_,
};
use crate::params::Param;
use crate::tensor::{raw_binary_k, raw_unary_k, Tensor};
use crate::Result;

/// Capture and reinstate everything an optimizer's update rule reads, so a
/// resumed run continues bit-exactly instead of warm-restarting moments.
/// Arrays are named per parameter by position ("m.0", "velocity.2", ...);
/// `Checkpoint` prefixes them with "optim." and stores them in
/// optimizer.safetensors alongside model.safetensors.
pub trait OptimizerState {
    fn snapshot(&self) -> Vec<(String, Tensor)>;
    /// Exact state keys accepted for a validated saved configuration. The
    /// default preserves strict snapshot-schema matching for custom optimizers.
    fn expected_restore_keys(&self, _config: &[f32]) -> Result<Vec<String>> {
        Ok(self.snapshot().into_iter().map(|(name, _)| name).collect())
    }
    /// Ordered, deduplicated parameter slots for checkpoint identity validation.
    fn state_parameters(&self) -> Option<Vec<Param>> { None }
    /// Live optimizer buffers for alias validation, not owning snapshots.
    fn state_buffers(&self) -> Option<Vec<Tensor>> { None }
    /// Prepare without live mutation. Dropping the token aborts; invoking it is
    /// infallible and preserves live buffer identities. CPU only.
    fn prepare_cpu_restore<'a>(&'a mut self, _config: &[f32], _state: &[(String, Tensor)]) -> Result<Box<dyn FnOnce() + 'a>> {
        Err(Error::Unsupported { op: "optim_restore", msg: "optimizer has no CPU transaction protocol".into() })
    }
    fn configuration(&self) -> Vec<f32> { Vec::new() }
    fn restore_configuration(&mut self, config: &[f32]) -> Result<()> {
        if !config.is_empty() { return Err(Error::Format { op: "optim_restore", msg: "unsupported configuration".into() }); }
        Ok(())
    }
    /// Strict restore: every array from `snapshot` must be present with
    /// matching shape, and nothing extra may remain. Validation/staging errors
    /// leave live state unchanged. Existing buffers retain storage and bump
    /// versions on commit; device commit errors may leave partial updates and
    /// require discarding the optimizer. This is not a model-state transaction.
    fn restore(&mut self, tensors: &[(String, Tensor)]) -> Result<()>;
}

// Raw kernels, not the ops.rs methods: optimizer math must never record an
// autograd node (params are grad-requiring leaves) and must dispatch straight
// to the device backend.
fn bmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    raw_binary_k("optim_mul", a, b, BinaryKind::Mul)
}
fn badd(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    raw_binary_k("optim_add", a, b, BinaryKind::Add)
}
fn bsub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    raw_binary_k("optim_sub", a, b, BinaryKind::Sub)
}
fn bdiv(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    raw_binary_k("optim_div", a, b, BinaryKind::Div)
}

fn cpu_amsgrad_step(parameter: &Tensor, mean: &Tensor, variance: &Tensor, maximum: &Tensor, gradient: &Tensor, hp: AdamWStep) -> Result<()> {
    let mut means = mean.to_vec(); let mut variances = variance.to_vec(); let mut maxima = maximum.to_vec(); let mut values = parameter.to_vec();
    let decay = (1.0 - hp.lr as f64 * hp.weight_decay as f64) as f32;
    for (j, g) in gradient.to_vec().into_iter().enumerate() {
        means[j] = hp.beta1 * means[j] + (1.0 - hp.beta1) * g;
        variances[j] = hp.beta2 * variances[j] + (1.0 - hp.beta2) * g * g;
        maxima[j] = if maxima[j].is_nan() || variances[j].is_nan() { f32::NAN } else { maxima[j].max(variances[j]) };
        if hp.weight_decay != 0.0 { values[j] *= decay; }
        values[j] -= hp.lr * (means[j] / hp.bc1) / ((maxima[j] / hp.bc2).sqrt() + hp.eps);
    }
    for (target, values) in [(mean, means), (variance, variances), (maximum, maxima), (parameter, values)] {
        crate::inplace::raw_copy_("adam_amsgrad", target, &Tensor::from_vec(values, parameter.shape())?)?;
    }
    Ok(())
}

/// A scalar as a tensor on `dev`, for the cold paths that still combine
/// tensors with scalars through binary kernels (the clip pre-scale and the
/// legacy fallback update): the fused in-place steps take scalars as plain
/// f32 kernel arguments and never touch this.
fn step_scalar(val: f32, dev: Device) -> Tensor {
    if dev == Device::Cpu {
        return Tensor::scalar(val);
    }
    Tensor::scalar(val)
        .to_device(dev)
        .expect("param's device backend is registered")
}

/// Lazily-initialized zero state buffer matching `t`'s shape and device.
fn zero_like(t: &Tensor) -> Tensor {
    Tensor::full_on(t.shape(), 0.0, t.device()).expect("param's device backend is registered")
}

fn cpu_step_gradient(gradient: Tensor) -> Result<Tensor> {
    if gradient.device() == Device::Cpu && (gradient.storage_offset() != 0 || !gradient.is_contiguous()
        || gradient.storage_len() != gradient.numel() || gradient.is_structural_zero()) {
        gradient.try_owned_detach_copy()
    } else { Ok(gradient) }
}

fn check_buf(
    slot: &mut Option<Tensor>,
    expected_shape: &[usize],
    key: &str,
    tensors: &[(String, Tensor)],
    dev: Device,
) -> Result<()> {
    let t = tensors
        .iter()
        .find(|(n, _)| n == key)
        .ok_or_else(|| Error::Format {
            op: "optim_restore",
            msg: format!("optimizer state is missing {key:?}"),
        })?;
    // Snapshot arrays are stored flat (one element per parameter); validate
    // against the element count, then bring back to the parameter's layout.
    let n: usize = expected_shape.iter().product();
    if t.1.shape() != [n] || t.1.dtype() != crate::dtype::DType::F32 {
        return Err(Error::Format {
            op: "optim_restore",
            msg: format!(
                "{key:?}: expected flat f32 [{n}] (for {expected_shape:?}), got {:?} {:?}",
                t.1.dtype(),
                t.1.shape()
            ),
        });
    }
    // State goes back to the parameter's device. A freshly constructed optimiser
    // has empty (None) slots, so we cannot infer the device from the slot -- the
    // caller passes the param's device so restored moments land where the
    // (capturable) update expects them, not defaulting to CPU.
    let up = t.1.owned_detach_copy().to_device(dev)?;
    *slot = Some(up.reshape(expected_shape)?);
    Ok(())
}

/// Stochastic gradient descent with optional heavy-ball momentum, nesterov
/// lookahead, and global-norm gradient clipping.
/// Cloning shares parameter slots but owns independent optimizer buffers.
/// Copying populated device buffers may transfer through host memory.
pub struct Sgd {
    precise: Option<[f64; 5]>,
    max_grad_norm_f64: bool,
    params: Vec<Param>,
    lr: f32,
    momentum: f32,
    nesterov: bool,
    weight_decay: f32,
    dampening: f32,
    maximize: bool,
    initialized: Vec<bool>,
    max_grad_norm: Option<f32>,
    velocity: Vec<Option<Tensor>>,
}

impl Clone for Sgd {
    fn clone(&self) -> Self {
        let mut cloned = self.clone_for_restore();
        cloned.velocity = self.velocity.iter().map(|t| t.as_ref().map(Tensor::owned_detach_copy)).collect();
        cloned
    }
}

impl Sgd {
    fn clone_for_restore(&self) -> Self {
        Self { precise: self.precise, max_grad_norm_f64: self.max_grad_norm_f64, params: self.params.clone(), lr: self.lr, momentum: self.momentum, nesterov: self.nesterov,
            weight_decay: self.weight_decay, dampening: self.dampening, maximize: self.maximize, initialized: self.initialized.clone(),
            max_grad_norm: self.max_grad_norm, velocity: self.velocity.clone() }
    }

    pub fn new(params: Vec<Param>, lr: f32) -> Sgd {
        let mut seen = std::collections::HashSet::new();
        let params: Vec<_> = params.into_iter().filter(|p| seen.insert(p.identity())).collect();
        let velocity = params.iter().map(|_| None).collect();
        let initialized = vec![false; params.len()];
        let precise = params.iter().any(|p| p.tensor().dtype() == crate::DType::F64).then_some([lr as f64,0.,0.,0.,0.]);
        Sgd {
            precise, max_grad_norm_f64: false,
            params,
            lr,
            momentum: 0.0,
            nesterov: false,
            weight_decay: 0.0,
            dampening: 0.0,
            maximize: false,
            initialized,
            max_grad_norm: None,
            velocity,
        }
    }

    pub fn with_momentum(mut self, m: f32) -> Sgd {
        self.momentum = m;
        if let Some(values) = &mut self.precise { values[1] = m as f64; }
        self
    }

    pub fn with_nesterov(mut self, on: bool) -> Sgd {
        self.nesterov = on;
        self
    }

    pub fn with_weight_decay(mut self, decay: f32) -> Sgd { self.weight_decay = decay; if let Some(v) = &mut self.precise { v[2] = decay as f64; } self }
    pub fn with_dampening(mut self, dampening: f32) -> Sgd { self.dampening = dampening; if let Some(v) = &mut self.precise { v[3] = dampening as f64; } self }
    pub fn with_maximize(mut self, maximize: bool) -> Sgd { self.maximize = maximize; self }

    pub fn with_max_grad_norm(mut self, max_norm: f32) -> Sgd {
        self.max_grad_norm = Some(max_norm);
        if let Some(v) = &mut self.precise { v[4] = max_norm as f64; self.max_grad_norm_f64 = true; }
        self
    }

    /// The learning rate the next step will use.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set the learning rate for subsequent steps: the seam an
    /// `LrScheduler` drives, `opt.set_lr(sched.lr(step))` before each step.
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        if let Some(v) = &mut self.precise { v[0] = lr as f64; }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        // Kernel dispatch on same-shape f32 buffers cannot fail; the Result
        // exists for the fallible broadcast path optimizers never take.
        self.update().expect("sgd step");
    }

    fn update(&mut self) -> Result<()> {
        validate_gradient_storage(&self.params)?;
        self.validate_parameters()?;
        if self.dampening != 0.0 {
            if crate::capture::is_recording() { return Err(Error::Unsupported { op: "sgd_step", msg: "dampened SGD requires eager CPU execution".into() }); }
            validate_cpu_parameters(&self.params)?;
            validate_cpu_buffers(&self.velocity)?;
            if self.nesterov { return Err(Error::InvalidShape { op: "sgd_step", msg: "Nesterov requires zero dampening".into() }); }
        }
        let has_f64 = self.params.iter().any(|p| p.tensor().dtype() == crate::DType::F64);
        let f64_scale = if has_f64 { self.f64_clip_scale() } else { 1. };
        let scale = if has_f64 { f64_scale as f32 } else { clip_scale(&self.params, self.max_grad_norm) };
        for i in 0..self.params.len() {
            let p = self.params[i].clone();
            let Some(g) = p.grad() else { continue };
            if p.tensor().dtype() == crate::DType::F64 { self.update_f64_slot(i, &g, f64_scale)?; continue; }
            let g = if self.momentum != 0.0 { cpu_step_gradient(g)? } else { g };
            let cur = p.tensor();
            let dev = cur.device();
            let mut g = if scale != 1.0 {
                bmul(&g, &step_scalar(scale, dev))?
            } else {
                g
            };
            if self.maximize { g = bmul(&g, &step_scalar(-1.0, dev))?; }
            if self.weight_decay != 0.0 { g = badd(&g, &bmul(&cur, &step_scalar(self.weight_decay, dev))?)?; }
            if self.dampening != 0.0 && self.momentum != 0.0 {
                let gradient = g.to_vec();
                let previous = self.velocity[i].as_ref().map(Tensor::to_vec).unwrap_or_else(|| vec![0.0; cur.numel()]);
                let next: Vec<_> = gradient.iter().zip(previous).map(|(&g, old)| {
                    if self.initialized[i] { self.momentum * old + (1.0 - self.dampening) * g } else { g }
                }).collect();
                let values: Vec<_> = cur.to_vec().into_iter().zip(&next).map(|(p, v)| p - self.lr * v).collect();
                let mut pending = [Some(Tensor::from_vec(next, cur.shape())?)];
                commit_buffers(std::slice::from_ref(&self.velocity[i]), &mut pending)?;
                crate::inplace::raw_copy_("sgd_step", &cur, &Tensor::from_vec(values, cur.shape())?)?;
                self.velocity[i] = pending[0].take();
                self.initialized[i] = true;
                p.zero_grad();
                continue;
            }
            // Fused in-place step: parameter and velocity keep their storage.
            let stepped = if self.momentum == 0.0 {
                raw_axpy_("sgd_step", -self.lr, &cur, &g).is_ok()
            } else {
                let v = self.velocity[i].get_or_insert_with(|| zero_like(&cur));
                raw_sgd_step_(&cur, v, &g, self.lr, self.momentum, self.nesterov).is_ok()
            };
            if stepped {
                if self.momentum != 0.0 { self.initialized[i] = true; }
                // The old step replaced the leaf, implicitly dropping its
                // grad; the in-place step keeps the tensor, so consume the
                // grad explicitly - silent cross-step accumulation is the
                // torch footgun this engine refuses to inherit.
                p.zero_grad();
                continue;
            }
            // Fallback (strided/aliased param, backend without in-place
            // kernels): the original allocating update, identical numbers.
            let d = if self.momentum == 0.0 {
                g
            } else {
                let mom = step_scalar(self.momentum, dev);
                let v = self.velocity[i].get_or_insert_with(|| zero_like(&cur));
                let nv = badd(&bmul(v, &mom)?, &g)?;
                let d = if self.nesterov {
                    badd(&bmul(&nv, &mom)?, &g)?
                } else {
                    nv.clone()
                };
                *v = nv;
                d
            };
            let lr_s = step_scalar(self.lr, dev);
            p.set(bsub(&cur, &bmul(&d, &lr_s)?)?);
            if self.momentum != 0.0 { self.initialized[i] = true; }
        }
        Ok(())
    }
}

impl OptimizerState for Sgd {
    fn expected_restore_keys(&self, config: &[f32]) -> Result<Vec<String>> {
        let mut staged = self.clone_for_restore();
        staged.restore_configuration(config)?;
        Ok((0..self.params.len()).flat_map(|i| { let mut keys = vec![format!("velocity.{i}")]; if staged.has_initialization_flags() { keys.push(format!("initialized.{i}")); } keys }).collect())
    }
    fn state_buffers(&self) -> Option<Vec<Tensor>> { Some(self.velocity.iter().flatten().cloned().collect()) }
    fn state_parameters(&self) -> Option<Vec<Param>> { Some(self.params.clone()) }
    fn prepare_cpu_restore<'a>(&'a mut self, config: &[f32], state: &[(String, Tensor)]) -> Result<Box<dyn FnOnce() + 'a>> {
        self.validate_parameters()?;
        validate_cpu_parameters(&self.params)?;
        validate_cpu_buffers(&self.velocity)?;
        let mut names = std::collections::HashSet::new();
        if state.iter().any(|(n,_)| !names.insert(n)) { return Err(Error::Format { op: "optim_restore", msg: "duplicate optimizer key".into() }); }
        let mut staged = self.clone_for_restore();
        staged.restore_configuration(config)?;
        staged.restore_staged(state)?;
        validate_buffer_copies(&self.velocity, &staged.velocity)?;
        Ok(Box::new(move || {
            commit_buffers(&self.velocity, &mut staged.velocity).expect("validated CPU buffers");
            *self = staged;
        }))
    }
    fn configuration(&self) -> Vec<f32> {
        if let Some(config) = self.precise_configuration() { return config; }
        let mut config = vec![1.0, self.lr, self.momentum, self.nesterov as u8 as f32, self.max_grad_norm.unwrap_or(-1.0)];
        if self.weight_decay != 0.0 || self.dampening != 0.0 || self.maximize { config.extend([self.weight_decay, self.dampening, self.maximize as u8 as f32]); }
        config
    }
    fn restore_configuration(&mut self, c: &[f32]) -> Result<()> {
        if c.len() == 25 { return self.restore_precise_configuration(c); }
        if ![5,8].contains(&c.len()) || c[0] != 1.0 || c.iter().any(|x| !x.is_finite())
            || c[1] < 0.0 || c[2] < 0.0 || ![0.0, 1.0].contains(&c[3])
            || (c[3] == 1.0 && c[2] <= 0.0) || (c[4] != -1.0 && c[4] < 0.0)
            || (c.len() == 8 && (c[5] < 0.0 || c[6] < 0.0 || ![0.0,1.0].contains(&c[7]) || (c[3] == 1.0 && c[6] != 0.0))) {
            return Err(Error::Format { op: "optim_restore", msg: "invalid optimizer configuration".into() });
        }
        self.precise = None; self.max_grad_norm_f64 = false;
        self.lr = c[1];
        self.momentum = c[2];
        self.nesterov = c[3] == 1.0;
        self.weight_decay = if c.len() == 8 { c[5] } else { 0.0 };
        self.dampening = if c.len() == 8 { c[6] } else { 0.0 };
        self.maximize = c.len() == 8 && c[7] == 1.0;
        self.max_grad_norm = if c[4] == -1.0 { None } else { Some(c[4]) };
        Ok(())
    }
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        // Cold path: state comes back to the host for serialization.
        let mut result: Vec<_> = self.velocity
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let host = v
                    .as_ref()
                    .map(|t| t.to_device(Device::Cpu).expect("cpu is always registered").owned_detach_copy())
                    .unwrap_or_else(|| { let p = self.params[i].tensor(); if p.dtype() == crate::DType::F64 {
                        Tensor::from_vec_f64(vec![0.;p.numel()], &[p.numel()]).expect("validated parameter")
                    } else { zero_like_cpu(p.numel()) } });
                // State is stored flat (one element per parameter), matching
                // the pre-device file format.
                let n = host.numel();
                (
                    format!("velocity.{i}"),
                    host.reshape(&[n]).expect("flat reshape"),
                )
            })
            .collect();
        if self.has_initialization_flags() {
            result.extend(self.initialized.iter().enumerate().map(|(i, flag)| (format!("initialized.{i}"), Tensor::scalar(*flag as u8 as f32))));
        }
        result
    }

    fn restore(&mut self, tensors: &[(String, Tensor)]) -> Result<()> {

        let mut names = std::collections::HashSet::new();
        if tensors.iter().any(|(n,_)| !names.insert(n)) { return Err(Error::Format { op: "optim_restore", msg: "duplicate optimizer key".into() }); }
        let mut staged = self.clone_for_restore();
        staged.restore_staged(tensors)?;
        validate_buffer_copies(&self.velocity, &staged.velocity)?;
        commit_buffers(&self.velocity, &mut staged.velocity)?;
        *self = staged;
        Ok(())
    }
}

impl Sgd {
    fn restore_staged(&mut self, tensors: &[(String, Tensor)]) -> Result<()> {
        self.validate_parameters()?;
        let expected = self.velocity.len() * if self.has_initialization_flags() { 2 } else { 1 };
        if tensors.len() != expected {
            return Err(Error::Format {
                op: "optim_restore",
                msg: format!(
                    "expected {} velocity buffers, got {}",
                    expected,
                    tensors.len()
                ),
            });
        }
        for i in 0..self.velocity.len() {
            let dev = self.params[i].tensor().device();
            let parameter = self.params[i].tensor();
            let key = format!("velocity.{i}");
            if parameter.dtype() == crate::DType::F64 {
                let value = &tensors.iter().find(|(name,_)| name == &key).ok_or_else(|| Error::Format { op: "optim_restore", msg: format!("missing {key}") })?.1;
                if value.dtype() != crate::DType::F64 || value.device() != Device::Cpu || value.shape() != [parameter.numel()] {
                    return Err(Error::Format { op: "optim_restore", msg: "SGD momentum dtype/shape/device mismatch".into() });
                }
                self.velocity[i] = Some(value.try_owned_detach_copy()?.reshape(parameter.shape())?);
            } else { check_buf(&mut self.velocity[i],parameter.shape(),&key,tensors,dev)?; }
            self.initialized[i] = if self.has_initialization_flags() {
                let flag = tensors.iter().find(|(name, _)| name == &format!("initialized.{i}")).ok_or_else(|| Error::Format { op: "optim_restore", msg: "missing SGD initialized flag".into() })?;
                if flag.1.dtype() != crate::dtype::DType::F32 || flag.1.device() != Device::Cpu || flag.1.shape() != [] || ![0.0,1.0].contains(&flag.1.item()) {
                    return Err(Error::Format { op: "optim_restore", msg: "invalid SGD initialized flag".into() });
                }
                flag.1.item() == 1.0
            } else { true };
            if self.precise.is_some() && !self.initialized[i]
                && self.velocity[i].as_ref().unwrap().to_vec_f64().iter().any(|v| v.to_bits() != 0) {
                return Err(Error::Format { op: "optim_restore", msg: "uninitialized SGD momentum must contain positive zeros".into() });
            }
        }
        Ok(())
    }
}

fn validate_buffer_copies(old: &[Option<Tensor>], new: &[Option<Tensor>]) -> Result<()> {
    for (old, new) in old.iter().zip(new) {
        if let (Some(dst), Some(src)) = (old, new) {
            if dst.shape() != src.shape() || dst.dtype() != src.dtype() || dst.device() != src.device()
                || dst.0.offset != 0 || !dst.is_contiguous() || dst.numel() != dst.storage_len() {
                return Err(Error::Format { op: "optim_restore", msg: "stale optimizer buffer shape or layout".into() });
            }
        }
    }
    Ok(())
}

#[doc(hidden)]
pub fn validate_f32_parameters(params: &[Param]) -> Result<()> {
    if params.iter().any(|p| p.tensor().dtype() != crate::DType::F32) {
        return Err(Error::Unsupported { op: "optimizer", msg: "this optimizer requires float32 parameters; CPU float64 is supported by SGD, Adam and AdamW".into() });
    }
    Ok(())
}

#[doc(hidden)]
pub fn validate_adam_parameters(params: &[Param]) -> Result<()> { adam_f64::validate_real_parameters(params) }

fn validate_cpu_f32_parameters(params: &[Param]) -> Result<()> {
    validate_f32_parameters(params)?;
    validate_cpu_parameters(params)
}

fn validate_cpu_parameters(params: &[Param]) -> Result<()> {
    for p in params { crate::checkpoint::validate_destination(&p.tensor(), &p.tensor())?; }
    Ok(())
}

/// Manual gradients sharing parameter storage need per-family mutation ordering
/// that these optimizers do not implement. Check all groups together at adapters.
#[doc(hidden)]
pub fn validate_gradient_storage(params: &[Param]) -> Result<()> {
    if crate::forward_ad::is_active() {
        return Err(Error::Unsupported { op: "optimizer_step", msg: "optimizer mutation during forward AD is unsupported".into() });
    }
    let mut identities = std::collections::HashSet::new();
    if params.iter().any(|p| !identities.insert(p.tensor().id())) {
        return Err(Error::Unsupported { op: "optimizer_step", msg: "optimizer parameters must have distinct gradient slots".into() });
    }
    let storage: std::collections::HashSet<_> = params.iter().map(|p| p.tensor()._storage_ptr()).collect();
    if params.iter().filter_map(Param::grad).any(|g| storage.contains(&g._storage_ptr())) {
        return Err(Error::Unsupported { op: "optimizer_step", msg: "active gradients must not share parameter storage".into() });
    }
    Ok(())
}

// Distinct leaves may share whole storage. Pending updates must observe earlier
// slots in optimizer order without publishing writes before validation finishes.
#[derive(Default)]
struct StagedParameterValues(std::collections::HashMap<usize, Tensor>);
impl StagedParameterValues {
    fn read(&self, parameter: &Tensor, original: impl FnOnce() -> Vec<f32>) -> Vec<f32> {
        self.0.get(&parameter._storage_ptr()).map(Tensor::to_vec).unwrap_or_else(original)
    }
    fn read_f64(&self, parameter: &Tensor) -> Vec<f64> {
        self.0.get(&parameter._storage_ptr()).map(Tensor::to_vec_f64).unwrap_or_else(|| parameter.to_vec_f64())
    }
    fn stage(&mut self, parameter: &Tensor, next: &Tensor) {
        self.0.insert(parameter._storage_ptr(), next.clone());
    }
}

fn validate_cpu_buffers(buffers: &[Option<Tensor>]) -> Result<()> {
    for t in buffers.iter().flatten() { crate::checkpoint::validate_destination(t, t)?; }
    Ok(())
}

fn commit_buffers(old: &[Option<Tensor>], new: &mut [Option<Tensor>]) -> Result<()> {
    for (old, new) in old.iter().zip(new) {
        if let (Some(dst), Some(src)) = (old, new.as_ref()) {
            if let Err(e) = crate::inplace::raw_copy_("optim_restore", dst, src) {
                // A backend may have enqueued a write before reporting failure.
                dst.bump_version();
                return Err(Error::Unsupported { op: "optim_restore", msg: format!("state commit failed; buffers may be partially updated; backend rollback is unsupported: {e}") });
            }
            *new = Some(dst.clone());
        }
    }
    Ok(())
}

fn checked_step(t: &Tensor) -> Result<f32> {
    if t.shape() != [] || t.dtype() != crate::dtype::DType::F32 {
        return Err(Error::Format { op: "optim_restore", msg: "step must be scalar f32".into() });
    }
    let x = t.item();
    if !x.is_finite() || x < 0.0 || x.fract() != 0.0 || x as f64 > u32::MAX as f64 {
        return Err(Error::Format { op: "optim_restore", msg: "step out of range".into() });
    }
    Ok(x)
}

fn zero_like_cpu(len: usize) -> Tensor {
    Tensor::from_vec(vec![0.0; len], &[len]).expect("flat f32 buffer")
}

/// Adam with bias correction. Defaults: beta1=0.9, beta2=0.999, eps=1e-8.
/// Cloning shares parameter slots but owns independent optimizer buffers.
/// Copying populated device buffers may transfer through host memory.
pub struct Adam {
    params: Vec<Param>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
    maximize: bool,
    max_grad_norm: Option<f32>,
    precise: Option<[f64; 6]>,
    /// Per-parameter step counts: a param that skips a step (no grad) must not
    /// advance its bias correction, or its next update is scaled wrongly.
    t: Vec<u32>,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    vmax: Vec<Option<Tensor>>,
}

impl Clone for Adam {
    fn clone(&self) -> Self {
        let mut cloned = self.clone_for_restore();
        cloned.m = self.m.iter().map(|t| t.as_ref().map(Tensor::owned_detach_copy)).collect();
        cloned.v = self.v.iter().map(|t| t.as_ref().map(Tensor::owned_detach_copy)).collect();
        cloned.vmax = self.vmax.iter().map(|t| t.as_ref().map(Tensor::owned_detach_copy)).collect();
        cloned
    }
}

impl Adam {
    fn clone_for_restore(&self) -> Self {
        Self { params: self.params.clone(), lr: self.lr, beta1: self.beta1, beta2: self.beta2, eps: self.eps,
            weight_decay: self.weight_decay, amsgrad: self.amsgrad, maximize: self.maximize,
            max_grad_norm: self.max_grad_norm, precise: self.precise, t: self.t.clone(), m: self.m.clone(), v: self.v.clone(), vmax: self.vmax.clone() }
    }

    pub fn new(params: Vec<Param>, lr: f32) -> Adam {
        adam_f64::validate_real_parameters(&params).expect("Adam requires float32 or CPU float64 parameters");
        let mut seen = std::collections::HashSet::new();
        let params: Vec<_> = params.into_iter().filter(|p| seen.insert(p.identity())).collect();
        let m = params.iter().map(|_| None).collect();
        let v = params.iter().map(|_| None).collect();
        let vmax = params.iter().map(|_| None).collect();
        let t = vec![0u32; params.len()];
        let precise = params.iter().any(|p| p.tensor().dtype() == crate::DType::F64)
            .then_some([lr as f64, 0.9, 0.999, 1e-8, 0.0, 0.0]);
        Adam {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            maximize: false,
            max_grad_norm: None,
            precise,
            t,
            m,
            v,
            vmax,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Adam {
        self.beta1 = beta1;
        self.beta2 = beta2;
        if let Some(v) = &mut self.precise { v[1] = beta1 as f64; v[2] = beta2 as f64; }
        self
    }

    pub fn with_eps(mut self, eps: f32) -> Adam {
        self.eps = eps;
        if let Some(v) = &mut self.precise { v[3] = eps as f64; }
        self
    }

    pub fn with_weight_decay(mut self, decay: f32) -> Adam { self.weight_decay = decay; if let Some(v) = &mut self.precise { v[4] = decay as f64; } self }
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Adam { self.amsgrad = amsgrad; self }
    pub fn with_maximize(mut self, maximize: bool) -> Adam { self.maximize = maximize; self }

    pub fn with_max_grad_norm(mut self, max_norm: f32) -> Adam {
        self.max_grad_norm = Some(max_norm);
        if let Some(v) = &mut self.precise { v[5] = max_norm as f64; }
        self
    }

    /// The learning rate the next step will use.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set the learning rate for subsequent steps (`LrScheduler` seam).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        if let Some(v) = &mut self.precise { v[0] = lr as f64; }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        self.update().expect("adam step");
    }

    fn update(&mut self) -> Result<()> {
        validate_gradient_storage(&self.params)?;
        self.validate_parameters()?;
        if self.amsgrad {
            if crate::capture::is_recording() { return Err(Error::Unsupported { op: "adam_step", msg: "AMSGrad requires eager CPU execution".into() }); }
            validate_cpu_parameters(&self.params)?;
            validate_cpu_buffers(&self.m)?;
            validate_cpu_buffers(&self.v)?;
            validate_cpu_buffers(&self.vmax)?;
        }
        let scale64 = self.precise.map(|_| self.precise_clip_scale());
        let scale = scale64.map(|s| s as f32).unwrap_or_else(|| clip_scale(&self.params, self.max_grad_norm));
        for i in 0..self.params.len() {
            let Some(g) = self.params[i].grad() else {
                continue;
            };
            if self.params[i].tensor().dtype() == crate::DType::F64 {
                self.update_f64_slot(i, &g, scale64.unwrap_or(1.))?;
                continue;
            }
            let g = cpu_step_gradient(g)?;
            let p = &self.params[i];
            let cur = p.tensor();
            let dev = cur.device();
            self.t[i] += 1;
            let bc1 = 1.0 - self.beta1.powi(self.t[i] as i32);
            let bc2 = 1.0 - self.beta2.powi(self.t[i] as i32);
            let mut g = if scale != 1.0 {
                bmul(&g, &step_scalar(scale, dev))?
            } else {
                g
            };
            if self.maximize { g = bmul(&g, &step_scalar(-1.0, dev))?; }
            if self.weight_decay != 0.0 { g = badd(&g, &bmul(&cur, &step_scalar(self.weight_decay, dev))?)?; }
            let m = self.m[i].get_or_insert_with(|| zero_like(&cur));
            let v = self.v[i].get_or_insert_with(|| zero_like(&cur));
            if self.amsgrad {
                let maximum = self.vmax[i].get_or_insert_with(|| zero_like(&cur));
                cpu_amsgrad_step(&cur,m,v,maximum,&g,AdamWStep { lr: self.lr, beta1: self.beta1, beta2: self.beta2, bc1, bc2, eps: self.eps, weight_decay: 0.0 })?;
                p.zero_grad();
                continue;
            }
            // Fused in-place step: parameter and both moments keep their
            // storage; every scalar rides in as an f32 kernel argument.
            let hp = AdamWStep {
                lr: self.lr,
                beta1: self.beta1,
                beta2: self.beta2,
                bc1,
                bc2,
                eps: self.eps,
                weight_decay: 0.0,
            };
            if raw_adamw_step_(&cur, m, v, &g, hp).is_ok() {
                p.zero_grad(); // consume the grad, like the old leaf replace
                continue;
            }
            // Fallback: the original allocating update, identical numbers.
            let b1 = step_scalar(self.beta1, dev);
            let nb1 = step_scalar(1.0 - self.beta1, dev);
            let b2 = step_scalar(self.beta2, dev);
            let nb2 = step_scalar(1.0 - self.beta2, dev);
            let eps = step_scalar(self.eps, dev);
            let lr = step_scalar(self.lr, dev);
            let bc1 = step_scalar(bc1, dev);
            let bc2 = step_scalar(bc2, dev);
            let nm = badd(&bmul(m, &b1)?, &bmul(&g, &nb1)?)?;
            let nv = badd(&bmul(v, &b2)?, &bmul(&bmul(&g, &g)?, &nb2)?)?;
            *m = nm;
            *v = nv;
            let m_hat = bdiv(m, &bc1)?;
            let denom = raw_unary_k(&bdiv(v, &bc2)?, UnaryKind::Sqrt)?;
            let upd = bdiv(&m_hat, &badd(&denom, &eps)?)?;
            p.set(bsub(&cur, &bmul(&upd, &lr)?)?);
        }
        Ok(())
    }
}

impl OptimizerState for Adam {
    fn expected_restore_keys(&self, config: &[f32]) -> Result<Vec<String>> {
        let mut staged = self.clone_for_restore();
        staged.restore_configuration(config)?;
        Ok(staged.snapshot().into_iter().map(|(name, _)| name).collect())
    }
    fn state_buffers(&self) -> Option<Vec<Tensor>> { Some(self.m.iter().chain(&self.v).chain(&self.vmax).flatten().cloned().collect()) }
    fn state_parameters(&self) -> Option<Vec<Param>> { Some(self.params.clone()) }
    fn prepare_cpu_restore<'a>(&'a mut self, config: &[f32], state: &[(String, Tensor)]) -> Result<Box<dyn FnOnce() + 'a>> {
        validate_cpu_parameters(&self.params)?;
        validate_cpu_buffers(&self.m)?;
        validate_cpu_buffers(&self.v)?;
        validate_cpu_buffers(&self.vmax)?;
        let mut names = std::collections::HashSet::new();
        if state.iter().any(|(n,_)| !names.insert(n)) { return Err(Error::Format { op: "optim_restore", msg: "duplicate optimizer key".into() }); }
        let mut staged = self.clone_for_restore();
        staged.restore_configuration(config)?;
        staged.restore_staged(state)?;
        validate_buffer_copies(&self.m, &staged.m)?;
        validate_buffer_copies(&self.v, &staged.v)?;
        validate_buffer_copies(&self.vmax, &staged.vmax)?;
        Ok(Box::new(move || {
            commit_buffers(&self.m, &mut staged.m).expect("validated CPU buffers");
            commit_buffers(&self.v, &mut staged.v).expect("validated CPU buffers");
            commit_buffers(&self.vmax, &mut staged.vmax).expect("validated CPU buffers");
            *self = staged;
        }))
    }
    fn configuration(&self) -> Vec<f32> {
        if let Some(c) = self.precise_configuration() { return c; }
        let mut config = vec![2.0, self.lr, self.beta1, self.beta2, self.eps, self.max_grad_norm.unwrap_or(-1.0)];
        if self.weight_decay != 0.0 || self.amsgrad || self.maximize { config.extend([self.weight_decay, self.amsgrad as u8 as f32, self.maximize as u8 as f32]); }
        config
    }
    fn restore_configuration(&mut self, c: &[f32]) -> Result<()> {
        if c.len() == 29 { return self.restore_precise_configuration(c); }
        if ![6,9].contains(&c.len()) || c[0] != 2.0 || c.iter().any(|x| !x.is_finite())
            || c[1] < 0.0 || !(0.0..1.0).contains(&c[2]) || !(0.0..1.0).contains(&c[3])
            || c[4] <= 0.0 || (c[5] != -1.0 && c[5] < 0.0)
            || (c.len() == 9 && (c[6] < 0.0 || ![0.0,1.0].contains(&c[7]) || ![0.0,1.0].contains(&c[8]))) {
            return Err(Error::Format { op: "optim_restore", msg: "invalid optimizer configuration".into() });
        }
        self.lr = c[1];
        self.beta1 = c[2];
        self.beta2 = c[3];
        self.eps = c[4];
        self.weight_decay = if c.len() == 9 { c[6] } else { 0.0 };
        self.amsgrad = c.len() == 9 && c[7] == 1.0;
        self.maximize = c.len() == 9 && c[8] == 1.0;
        self.max_grad_norm = if c[5] == -1.0 { None } else { Some(c[5]) };
        self.precise = None;
        if self.params.iter().any(|p| p.tensor().dtype() == crate::DType::F64) { self.precise = Some(self.f64_values()); }
        Ok(())
    }
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        if self.precise.is_some() { return self.precise_snapshot(); }
        // Cold path: state comes back to the host for serialization.
        let mut out = Vec::new();
        for i in 0..self.params.len() {
            let host = |s: &Option<Tensor>| -> Tensor {
                let t = s
                    .as_ref()
                    .map(|t| t.to_device(Device::Cpu).expect("cpu is always registered").owned_detach_copy())
                    .unwrap_or_else(|| zero_like_cpu(self.params[i].tensor().numel()));
                // Stored flat (one element per parameter): file-format parity.
                let n = t.numel();
                t.reshape(&[n]).expect("flat reshape")
            };
            out.push((format!("m.{i}"), host(&self.m[i])));
            out.push((format!("v.{i}"), host(&self.v[i])));
            if self.amsgrad { out.push((format!("vmax.{i}"), host(&self.vmax[i]))); }
            out.push((format!("t.{i}"), Tensor::scalar(self.t[i] as f32)));
        }
        out
    }

    fn restore(&mut self, tensors: &[(String, Tensor)]) -> Result<()> {

        let mut names = std::collections::HashSet::new();
        if tensors.iter().any(|(n,_)| !names.insert(n)) { return Err(Error::Format { op: "optim_restore", msg: "duplicate optimizer key".into() }); }
        let mut staged = self.clone_for_restore();
        staged.restore_staged(tensors)?;
        validate_buffer_copies(&self.m, &staged.m)?;
        validate_buffer_copies(&self.v, &staged.v)?;
        validate_buffer_copies(&self.vmax, &staged.vmax)?;
        commit_buffers(&self.m, &mut staged.m)?;
        commit_buffers(&self.v, &mut staged.v)?;
        commit_buffers(&self.vmax, &mut staged.vmax)?;
        *self = staged;
        Ok(())
    }
}

impl Adam {
    fn restore_staged(&mut self, tensors: &[(String, Tensor)]) -> Result<()> {
        if self.precise.is_some() { return self.restore_precise_staged(tensors); }
        self.validate_parameters()?;
        let expected = (if self.amsgrad { 4 } else { 3 }) * self.params.len();
        if tensors.len() != expected {
            return Err(Error::Format {
                op: "optim_restore",
                msg: format!(
                    "expected {} arrays, got {}",
                    expected,
                    tensors.len()
                ),
            });
        }
        for i in 0..self.params.len() {
            let shape = self.params[i].tensor().shape().to_vec();
            let dev = self.params[i].tensor().device();
            check_buf(&mut self.m[i], &shape, &format!("m.{i}"), tensors, dev)?;
            check_buf(&mut self.v[i], &shape, &format!("v.{i}"), tensors, dev)?;
            if self.amsgrad { check_buf(&mut self.vmax[i], &shape, &format!("vmax.{i}"), tensors, dev)?; }
            else { self.vmax[i] = None; }
            let ts = tensors
                .iter()
                .find(|(n, _)| n == &format!("t.{i}"))
                .ok_or_else(|| Error::Format {
                    op: "optim_restore",
                    msg: format!("optimizer state is missing t.{i}"),
                })?;
            let tv = checked_step(&ts.1)?;
            if tv < 0.0 || tv.fract() != 0.0 || tv > u32::MAX as f32 {
                return Err(Error::Format {
                    op: "optim_restore",
                    msg: format!("t.{i} must be a non-negative integer, got {tv}"),
                });
            }
            self.t[i] = tv as u32;
        }
        Ok(())
    }
}

/// AdamW: Adam with decoupled weight decay (Loshchilov-Hutter) - the decay
/// term `lr * wd * param` is applied directly to the parameter instead of
/// being folded into the gradient, so it never enters the moment estimates.
/// Defaults match torch: beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01.
/// Eager bias correction advances only for parameters with a gradient.
/// Legacy snapshots store `t` as an f32 vector in parameter order; exact CPU
/// configurations use f64 counters to preserve the native u32 count. Legacy scalar `t`
/// restores expand the recorded global count; historical skips are unrecoverable.
/// Cloning shares parameter slots but owns independent optimizer buffers.
/// Copying populated device buffers may transfer through host memory.
pub struct AdamW {
    params: Vec<Param>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
    maximize: bool,
    max_grad_norm: Option<f32>,
    precise: Option<[f64; 6]>,
    t: Vec<u32>,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    vmax: Vec<Option<Tensor>>,
    /// When set, the step runs the CUDA-graph-capturable path: the timestep
    /// lives in a device tensor `[step, bc1, bc2]` and bias correction advances
    /// in-kernel, so a captured step replays correctly (mirrors PyTorch
    /// `capturable=True`). `None` = the ordinary host-timestep path.
    capturable_t: Option<Tensor>,
}

impl Clone for AdamW {
    fn clone(&self) -> Self {
        let mut cloned = self.clone_for_restore();
        cloned.m = self.m.iter().map(|t| t.as_ref().map(Tensor::owned_detach_copy)).collect();
        cloned.v = self.v.iter().map(|t| t.as_ref().map(Tensor::owned_detach_copy)).collect();
        cloned.vmax = self.vmax.iter().map(|t| t.as_ref().map(Tensor::owned_detach_copy)).collect();
        cloned.capturable_t = self.capturable_t.as_ref().map(Tensor::owned_detach_copy);
        cloned
    }
}

impl AdamW {
    fn clone_for_restore(&self) -> Self {
        Self { params: self.params.clone(), lr: self.lr, beta1: self.beta1, beta2: self.beta2, eps: self.eps,
            weight_decay: self.weight_decay, amsgrad: self.amsgrad, maximize: self.maximize, max_grad_norm: self.max_grad_norm, precise: self.precise, t: self.t.clone(),
            m: self.m.clone(), v: self.v.clone(), vmax: self.vmax.clone(), capturable_t: self.capturable_t.clone() }
    }

    pub fn new(params: Vec<Param>, lr: f32) -> AdamW {
        adam_f64::validate_real_parameters(&params).expect("Adam requires float32 or CPU float64 parameters");
        let mut seen = std::collections::HashSet::new();
        let params: Vec<_> = params.into_iter().filter(|p| seen.insert(p.identity())).collect();
        let m = params.iter().map(|_| None).collect();
        let v = params.iter().map(|_| None).collect();
        let vmax = params.iter().map(|_| None).collect();
        let t = vec![0; params.len()];
        let precise = params.iter().any(|p| p.tensor().dtype() == crate::DType::F64)
            .then_some([lr as f64, 0.9, 0.999, 1e-8, 0.01, 0.0]);
        AdamW {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            amsgrad: false,
            maximize: false,
            max_grad_norm: None,
            precise,
            t,
            m,
            v,
            vmax,
            capturable_t: None,
        }
    }

    /// Enable the CUDA-graph-capturable step. The timestep is moved onto the
    /// params' device as a `[step, bc1, bc2]` tensor and bias correction is
    /// advanced in-kernel, so a step recorded with `begin_step_capture` /
    /// `end_step_capture` replays with an advancing correction instead of a
    /// frozen one.
    ///
    /// Capturable mode engages **only** when every parameter lives on the same
    /// CUDA device with equal per-parameter counters. On CPU params, mixed-device
    /// sets, unequal counters, or an empty param list this is a no-op and the
    /// optimiser keeps the ordinary host path — `is_capturable()` then stays
    /// `false`. `lr` and betas are fixed for the life of a captured graph
    /// (re-capture to change them). Each capturable update requires all parameter
    /// gradients or none; a partial set is rejected before advancing the counter.
    pub fn capturable(mut self) -> AdamW {
        if self.amsgrad || self.maximize || self.precise.is_some() { return self; }
        if self.capturable_t.is_some() { return self; }
        let Some(dev) = self.params.first().map(|p| p.tensor().device()) else {
            return self; // no params: nothing to capture.
        };
        // Device-only path: CPU params must retain the host update.
        if !matches!(dev, Device::Cuda(_)) {
            return self;
        }
        // A single device timestep buffer serves all params, so they must all
        // sit on the same device; otherwise fall back to the host path.
        if self.params.iter().any(|p| p.tensor().device() != dev) {
            return self;
        }
        let step = self.t[0];
        if self.t.iter().any(|&t| t != step) { return self; }
        // Seed the device timestep from the CURRENT host counter so enabling
        // capture AFTER eager warm-up steps preserves the timestep (and thus the
        // bias correction that matches the already-matured moment buffers).
        // The first increment follows the common eager counter.
        let t0 = step as f32;
        let bc1 = 1.0 - self.beta1.powi(step as i32);
        let bc2 = 1.0 - self.beta2.powi(step as i32);
        match Tensor::from_vec(vec![t0, bc1, bc2], &[3]).and_then(|t| t.to_device(dev)) {
            Ok(t) => self.capturable_t = Some(t),
            Err(_) => return self, // allocation failed: stay on host path.
        }
        self
    }

    /// True when the capturable device-timestep path is active.
    pub fn is_capturable(&self) -> bool {
        self.capturable_t.is_some()
    }

    pub fn with_weight_decay(mut self, wd: f32) -> AdamW {
        self.weight_decay = wd;
        if let Some(v) = &mut self.precise { v[4] = wd as f64; }
        self
    }

    pub fn with_amsgrad(mut self, amsgrad: bool) -> AdamW { self.amsgrad = amsgrad; self }
    pub fn with_maximize(mut self, maximize: bool) -> AdamW { self.maximize = maximize; self }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> AdamW {
        self.beta1 = beta1;
        self.beta2 = beta2;
        if let Some(v) = &mut self.precise { v[1] = beta1 as f64; v[2] = beta2 as f64; }
        self
    }

    pub fn with_max_grad_norm(mut self, max_norm: f32) -> AdamW {
        self.max_grad_norm = Some(max_norm);
        if let Some(v) = &mut self.precise { v[5] = max_norm as f64; }
        self
    }

    /// The learning rate the next step will use.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set the learning rate for subsequent steps (`LrScheduler` seam).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        if let Some(v) = &mut self.precise { v[0] = lr as f64; }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        self.update().expect("adamw step");
    }

    fn update(&mut self) -> Result<()> {
        validate_gradient_storage(&self.params)?;
        self.validate_parameters()?;
        if (self.amsgrad || self.maximize) && self.capturable_t.is_some() {
            return Err(Error::Unsupported { op: "adamw_step", msg: "AMSGrad/maximize do not support capturable AdamW".into() });
        }
        if self.amsgrad {
            if crate::capture::is_recording() { return Err(Error::Unsupported { op: "adamw_step", msg: "AMSGrad requires eager CPU execution".into() }); }
            validate_cpu_parameters(&self.params)?;
            validate_cpu_buffers(&self.m)?; validate_cpu_buffers(&self.v)?; validate_cpu_buffers(&self.vmax)?;
        }
        if let Some(t_dev) = self.capturable_t.clone() {
            return self.update_capturable(&t_dev);
        }
        let scale64 = self.precise.map(|_| self.precise_clip_scale());
        let scale = scale64.map(|s| s as f32).unwrap_or_else(|| clip_scale(&self.params, self.max_grad_norm));
        for i in 0..self.params.len() {
            let Some(g) = self.params[i].grad() else {
                continue;
            };
            if self.params[i].tensor().dtype() == crate::DType::F64 {
                self.update_f64_slot(i, &g, scale64.unwrap_or(1.))?;
                continue;
            }
            let g = cpu_step_gradient(g)?;
            let p = &self.params[i];
            let cur = p.tensor();
            let dev = cur.device();
            self.t[i] += 1;
            let bc1 = 1.0 - self.beta1.powi(self.t[i] as i32);
            let bc2 = 1.0 - self.beta2.powi(self.t[i] as i32);
            let mut g = if scale != 1.0 {
                bmul(&g, &step_scalar(scale, dev))?
            } else {
                g
            };
            if self.maximize { g = bmul(&g, &step_scalar(-1.0, dev))?; }
            let m = self.m[i].get_or_insert_with(|| zero_like(&cur));
            let v = self.v[i].get_or_insert_with(|| zero_like(&cur));
            if self.amsgrad {
                let maximum = self.vmax[i].get_or_insert_with(|| zero_like(&cur));
                cpu_amsgrad_step(&cur,m,v,maximum,&g,AdamWStep { lr: self.lr, beta1: self.beta1, beta2: self.beta2, bc1, bc2, eps: self.eps, weight_decay: self.weight_decay })?;
                p.zero_grad();
                continue;
            }
            // Fused in-place step (decoupled decay reads the pre-update
            // parameter inside the kernel): stable storage, f32 scalar args.
            let hp = AdamWStep {
                lr: self.lr,
                beta1: self.beta1,
                beta2: self.beta2,
                bc1,
                bc2,
                eps: self.eps,
                weight_decay: self.weight_decay,
            };
            if raw_adamw_step_(&cur, m, v, &g, hp).is_ok() {
                p.zero_grad(); // consume the grad, like the old leaf replace
                continue;
            }
            // Fallback: the original allocating update, identical numbers.
            let b1 = step_scalar(self.beta1, dev);
            let nb1 = step_scalar(1.0 - self.beta1, dev);
            let b2 = step_scalar(self.beta2, dev);
            let nb2 = step_scalar(1.0 - self.beta2, dev);
            let eps = step_scalar(self.eps, dev);
            let lr = step_scalar(self.lr, dev);
            let wd = step_scalar(self.weight_decay, dev);
            let bc1 = step_scalar(bc1, dev);
            let bc2 = step_scalar(bc2, dev);
            let nm = badd(&bmul(m, &b1)?, &bmul(&g, &nb1)?)?;
            let nv = badd(&bmul(v, &b2)?, &bmul(&bmul(&g, &g)?, &nb2)?)?;
            *m = nm;
            *v = nv;
            // p -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p): the decay term
            // reads the pre-update parameter, matching the old elementwise
            // loop's use of vals[j] before the subtract.
            let m_hat = bdiv(m, &bc1)?;
            let denom = raw_unary_k(&bdiv(v, &bc2)?, UnaryKind::Sqrt)?;
            let upd = badd(&bdiv(&m_hat, &badd(&denom, &eps)?)?, &bmul(&cur, &wd)?)?;
            p.set(bsub(&cur, &bmul(&upd, &lr)?)?);
        }
        Ok(())
    }

    /// CUDA-graph-capturable step. The timestep tensor `t_dev = [step, bc1, bc2]`
    /// is advanced once on-device (recomputing bias correction in-kernel), then
    /// each param runs the capturable fused step reading `t_dev`. Everything is
    /// device-side and address-stable, so a step recorded between
    /// `begin_step_capture`/`end_step_capture` replays with an advancing
    /// correction. Params/state/grads mutate in place; no host per-step state.
    fn update_capturable(&mut self, t_dev: &Tensor) -> Result<()> {
        let active = self.params.iter().filter(|p| p.grad().is_some()).count();
        if active == 0 { return Ok(()); }
        if active != self.params.len() {
            return Err(Error::Unsupported { op: "adamw_capturable", msg: "capturable AdamW requires gradients for every parameter or none".into() });
        }
        if self.max_grad_norm.is_some() {
            return Err(Error::Unsupported {
                op: "adamw_capturable",
                msg: "gradient clipping needs a host-side norm each step, which \
                      cannot be captured; disable max_grad_norm for capturable AdamW"
                    .to_string(),
            });
        }
        // One device increment per optimiser step: advances `step` and
        // recomputes bc1/bc2 in the timestep buffer, recorded as a graph node.
        // NOTE: under graph capture this only RECORDS; the buffer advances when
        // the graph replays. We deliberately do NOT touch the host `self.t`
        // here — the device buffer is the single source of truth for the
        // timestep (see snapshot/restore), so a captured step that replays N
        // times leaves `step` at N without this Rust code re-running.
        raw_scalar_increment_(t_dev, self.beta1, self.beta2)?;
        for i in 0..self.params.len() {
            let Some(g) = self.params[i].grad() else {
                continue;
            };
            let p = &self.params[i];
            let cur = p.tensor();
            let m = self.m[i].get_or_insert_with(|| zero_like(&cur));
            let v = self.v[i].get_or_insert_with(|| zero_like(&cur));
            raw_adamw_step_capturable_(
                &cur,
                m,
                v,
                &g,
                t_dev,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
            )?;
            p.zero_grad();
        }
        Ok(())
    }
}

impl OptimizerState for AdamW {
    fn expected_restore_keys(&self, config: &[f32]) -> Result<Vec<String>> {
        let mut staged = self.clone_for_restore();
        staged.restore_configuration(config)?;
        Ok(staged.snapshot().into_iter().map(|(name, _)| name).collect())
    }
    fn state_buffers(&self) -> Option<Vec<Tensor>> { Some(self.m.iter().chain(&self.v).chain(&self.vmax).flatten().cloned().chain(self.capturable_t.iter().cloned()).collect()) }
    fn state_parameters(&self) -> Option<Vec<Param>> { Some(self.params.clone()) }
    fn prepare_cpu_restore<'a>(&'a mut self, config: &[f32], state: &[(String, Tensor)]) -> Result<Box<dyn FnOnce() + 'a>> {
        validate_cpu_parameters(&self.params)?;
        validate_cpu_buffers(&self.m)?;
        validate_cpu_buffers(&self.v)?;
        validate_cpu_buffers(&self.vmax)?;
        validate_cpu_buffers(std::slice::from_ref(&self.capturable_t))?;
        let mut names = std::collections::HashSet::new();
        if state.iter().any(|(n,_)| !names.insert(n)) { return Err(Error::Format { op: "optim_restore", msg: "duplicate optimizer key".into() }); }
        let mut staged = self.clone_for_restore();
        staged.restore_configuration(config)?;
        staged.restore_staged(state)?;
        validate_buffer_copies(&self.m, &staged.m)?;
        validate_buffer_copies(&self.v, &staged.v)?;
        validate_buffer_copies(&self.vmax, &staged.vmax)?;
        validate_buffer_copies(std::slice::from_ref(&self.capturable_t), std::slice::from_ref(&staged.capturable_t))?;
        Ok(Box::new(move || {
            commit_buffers(&self.m, &mut staged.m).expect("validated CPU buffers");
            commit_buffers(&self.v, &mut staged.v).expect("validated CPU buffers");
            commit_buffers(&self.vmax, &mut staged.vmax).expect("validated CPU buffers");
            commit_buffers(std::slice::from_ref(&self.capturable_t), std::slice::from_mut(&mut staged.capturable_t)).expect("validated CPU buffers");
            *self = staged;
        }))
    }
    fn configuration(&self) -> Vec<f32> {
        if let Some(c) = self.precise_configuration() { return c; }
        let mut config = vec![3.0, self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self.max_grad_norm.unwrap_or(-1.0)];
        if self.amsgrad || self.maximize { config.extend([self.amsgrad as u8 as f32, self.maximize as u8 as f32]); }
        config
    }
    fn restore_configuration(&mut self, c: &[f32]) -> Result<()> {
        if c.len() == 29 { return self.restore_precise_configuration(c); }
        if ![7,9].contains(&c.len()) || c[0] != 3.0 || c.iter().any(|x| !x.is_finite())
            || c[1] < 0.0 || !(0.0..1.0).contains(&c[2]) || !(0.0..1.0).contains(&c[3])
            || c[4] <= 0.0 || c[5] < 0.0 || (c[6] != -1.0 && c[6] < 0.0)
            || (c.len() == 9 && (![0.0,1.0].contains(&c[7]) || ![0.0,1.0].contains(&c[8]))) {
            return Err(Error::Format { op: "optim_restore", msg: "invalid optimizer configuration".into() });
        }
        self.lr = c[1];
        self.beta1 = c[2];
        self.beta2 = c[3];
        self.eps = c[4];
        self.weight_decay = c[5];
        self.amsgrad = c.len() == 9 && c[7] == 1.0;
        self.maximize = c.len() == 9 && c[8] == 1.0;
        self.max_grad_norm = if c[6] == -1.0 { None } else { Some(c[6]) };
        self.precise = None;
        if self.params.iter().any(|p| p.tensor().dtype() == crate::DType::F64) { self.precise = Some(self.f64_values()); }
        Ok(())
    }
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        if self.precise.is_some() { return self.precise_snapshot(); }
        // Cold path: state comes back to the host for serialization.
        // In capturable mode the device timestep buffer is authoritative
        // (the host `self.t` is not advanced by replays), so read `step` back
        // from it; otherwise the host counter is the source of truth.
        let steps = match &self.capturable_t {
            Some(t) => vec![t.to_device(Device::Cpu).expect("cpu is always registered").to_vec()[0]; self.params.len()],
            None => self.t.iter().map(|&t| t as f32).collect(),
        };
        let mut out = vec![("t".to_string(), Tensor::from_vec(steps, &[self.params.len()]).expect("per-parameter steps"))];
        let host = |s: &Option<Tensor>, i: usize| -> Tensor {
            let t = s
                .as_ref()
                .map(|t| t.to_device(Device::Cpu).expect("cpu is always registered").owned_detach_copy())
                .unwrap_or_else(|| zero_like_cpu(self.params[i].tensor().numel()));
            // Stored flat (one element per parameter): file-format parity.
            let n = t.numel();
            t.reshape(&[n]).expect("flat reshape")
        };
        for i in 0..self.params.len() {
            out.push((format!("m.{i}"), host(&self.m[i], i)));
            out.push((format!("v.{i}"), host(&self.v[i], i)));
            if self.amsgrad { out.push((format!("vmax.{i}"), host(&self.vmax[i], i))); }
        }
        out
    }

    fn restore(&mut self, tensors: &[(String, Tensor)]) -> Result<()> {

        let mut names = std::collections::HashSet::new();
        if tensors.iter().any(|(n,_)| !names.insert(n)) { return Err(Error::Format { op: "optim_restore", msg: "duplicate optimizer key".into() }); }
        let mut staged = self.clone_for_restore();
        staged.restore_staged(tensors)?;
        validate_buffer_copies(&self.m, &staged.m)?;
        validate_buffer_copies(&self.v, &staged.v)?;
        validate_buffer_copies(&self.vmax, &staged.vmax)?;
        validate_buffer_copies(std::slice::from_ref(&self.capturable_t), std::slice::from_ref(&staged.capturable_t))?;
        commit_buffers(&self.m, &mut staged.m)?;
        commit_buffers(&self.v, &mut staged.v)?;
        commit_buffers(&self.vmax, &mut staged.vmax)?;
        commit_buffers(std::slice::from_ref(&self.capturable_t), std::slice::from_mut(&mut staged.capturable_t))?;
        *self = staged;
        Ok(())
    }
}

impl AdamW {
    fn restore_staged(&mut self, tensors: &[(String, Tensor)]) -> Result<()> {
        if self.precise.is_some() { return self.restore_precise_staged(tensors); }
        self.validate_parameters()?;
        if self.capturable_t.is_some() && (self.amsgrad || self.maximize) {
            return Err(Error::Format { op: "optim_restore", msg: "capturable AdamW does not support AMSGrad or maximize".into() });
        }
        let expected = (2 + self.amsgrad as usize) * self.params.len() + 1;
        if tensors.len() != expected {
            return Err(Error::Format {
                op: "optim_restore",
                msg: format!(
                    "expected {} arrays, got {}",
                    expected,
                    tensors.len()
                ),
            });
        }
        let ts = tensors
            .iter()
            .find(|(n, _)| n == "t")
            .ok_or_else(|| Error::Format {
                op: "optim_restore",
                msg: "optimizer state is missing t".into(),
            })?;
        let steps = if ts.1.shape() == [] {
            // Legacy snapshots recorded one global count; historical skipped
            // counts cannot be recovered, so retain that count for every slot.
            vec![checked_step(&ts.1)? as u32; self.params.len()]
        } else {
            if ts.1.shape() != [self.params.len()] || ts.1.dtype() != crate::DType::F32 {
                return Err(Error::Format { op: "optim_restore", msg: "AdamW t must be per-parameter f32 [n] or a legacy scalar".into() });
            }
            let values = ts.1.to_vec();
            if values.iter().any(|&x| !x.is_finite() || x < 0.0 || x.fract() != 0.0 || x as f64 > u32::MAX as f64) {
                return Err(Error::Format { op: "optim_restore", msg: "step out of range".into() });
            }
            values.into_iter().map(|x| x as u32).collect()
        };
        if self.capturable_t.is_some() && steps.iter().any(|&t| t != steps[0]) {
            return Err(Error::Format { op: "optim_restore", msg: "capturable AdamW requires equal per-parameter counters".into() });
        }
        self.t = steps;
        // In capturable mode the device buffer is authoritative, so push the
        // restored timestep + its bias correction back onto the device;
        // otherwise a resumed run would replay from step 0 (frozen correction).
        if self.capturable_t.is_some() {
            let step = self.t[0];
            let bc1 = 1.0 - self.beta1.powi(step as i32);
            let bc2 = 1.0 - self.beta2.powi(step as i32);
            let dev = self.params[0].tensor().device();
            let restored =
                Tensor::from_vec(vec![step as f32, bc1, bc2], &[3]).and_then(|t| t.to_device(dev))?;
            self.capturable_t = Some(restored);
        }
        for i in 0..self.params.len() {
            let shape = self.params[i].tensor().shape().to_vec();
            let dev = self.params[i].tensor().device();
            check_buf(&mut self.m[i], &shape, &format!("m.{i}"), tensors, dev)?;
            check_buf(&mut self.v[i], &shape, &format!("v.{i}"), tensors, dev)?;
            if self.amsgrad {
                check_buf(&mut self.vmax[i], &shape, &format!("vmax.{i}"), tensors, dev)?;
            } else {
                self.vmax[i] = None;
            }
        }
        Ok(())
    }
}

// --- gradient clipping ----------------------------------------------------

/// Global L2 norm over every parameter's accumulated gradient (params without
/// a grad contribute nothing). The reduction runs on the grads' device; the
/// one scalar read back is inherent - clipping branches on the norm value.
pub fn global_grad_norm(params: &[Param]) -> f32 {
    let mut acc = 0.0f32;
    for p in params {
        if let Some(g) = p.grad() {
            let sq = raw_binary_k("optim_mul", &g, &g, BinaryKind::Mul)
                .expect("grad is f32 on a registered device");
            acc += sq.sum().item();
        }
    }
    acc.sqrt()
}

/// The multiplicative factor applied when clipping to `max` (None or already
/// under the budget gives 1.0). Opt-in: enabling it costs one scalar read per
/// grad per step, since the clip decision needs the norm on the host.
fn clip_scale(params: &[Param], max: Option<f32>) -> f32 {
    match max {
        None => 1.0,
        Some(max) => {
            let norm = global_grad_norm(params);
            if norm > max && norm > 0.0 {
                max / norm
            } else {
                1.0
            }
        }
    }
}

// --- LR schedulers ---------------------------------------------------------

/// Learning rate as a closed-form function of the optimizer step count
/// (0-indexed; `lr(step)` is the lr to use FOR that step).
pub trait LrScheduler {
    fn lr(&self, step: usize) -> f32;
}

/// Halve-style decay: base_lr * gamma^floor(step / step_size), torch's
/// StepLR.
pub struct StepLr {
    pub base_lr: f32,
    pub step_size: usize,
    pub gamma: f32,
}

impl LrScheduler for StepLr {
    fn lr(&self, step: usize) -> f32 {
        self.base_lr * self.gamma.powi((step / self.step_size) as i32)
    }
}

/// Exponential decay: base_lr * gamma^step, torch's ExponentialLR.
pub struct ExponentialLr {
    pub base_lr: f32,
    pub gamma: f32,
}

impl LrScheduler for ExponentialLr {
    fn lr(&self, step: usize) -> f32 {
        self.base_lr * self.gamma.powi(step as i32)
    }
}

/// Linear warmup from ~0 to base_lr over `warmup_steps`, then cosine annealing
/// down to min_lr at `total_steps` (held there afterwards), torch's
/// get_cosine_schedule_with_warmup extended with a floor:
///
/// step < warmup:  base_lr * step / warmup
/// otherwise:      min_lr + 0.5*(base_lr-min_lr)*(1+cos(pi*(step-warmup)/(total-warmup)))
#[derive(Clone, Copy)]
pub struct CosineWithWarmup {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LrScheduler for CosineWithWarmup {
    fn lr(&self, step: usize) -> f32 {
        if self.total_steps <= self.warmup_steps {
            return self.base_lr;
        }
        if step < self.warmup_steps {
            return self.base_lr * step as f32 / self.warmup_steps as f32;
        }
        let progress = ((step - self.warmup_steps) as f32
            / (self.total_steps - self.warmup_steps) as f32)
            .min(1.0);
        self.min_lr
            + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}
