//! Optimizers (SGD, Adam, AdamW), LR schedulers, and gradient clipping,
//! operating on parameter tensors and their `.grad()`.
//!
//! Steps mutate IN PLACE through the fused no-grad seams in `inplace`
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
    /// Strict restore: every array from `snapshot` must be present with
    /// matching shape, and nothing extra may remain.
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

fn check_buf(
    slot: &mut Option<Tensor>,
    expected_shape: &[usize],
    key: &str,
    tensors: &[(String, Tensor)],
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
    // State goes back to wherever the parameters live; restore is a cold path.
    let dev = slot.as_ref().map(|s| s.device()).unwrap_or(Device::Cpu);
    let up = t.1.to_device(dev)?;
    *slot = Some(up.reshape(expected_shape)?);
    Ok(())
}

/// Stochastic gradient descent with optional heavy-ball momentum, nesterov
/// lookahead, and global-norm gradient clipping.
pub struct Sgd {
    params: Vec<Param>,
    lr: f32,
    momentum: f32,
    nesterov: bool,
    max_grad_norm: Option<f32>,
    velocity: Vec<Option<Tensor>>,
}

impl Sgd {
    pub fn new(params: Vec<Param>, lr: f32) -> Sgd {
        let velocity = params.iter().map(|_| None).collect();
        Sgd {
            params,
            lr,
            momentum: 0.0,
            nesterov: false,
            max_grad_norm: None,
            velocity,
        }
    }

    pub fn with_momentum(mut self, m: f32) -> Sgd {
        self.momentum = m;
        self
    }

    pub fn with_nesterov(mut self, on: bool) -> Sgd {
        self.nesterov = on;
        self
    }

    pub fn with_max_grad_norm(mut self, max_norm: f32) -> Sgd {
        self.max_grad_norm = Some(max_norm);
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
        let scale = clip_scale(&self.params, self.max_grad_norm);
        for (i, p) in self.params.iter().enumerate() {
            let Some(g) = p.grad() else { continue };
            let cur = p.tensor();
            let dev = cur.device();
            let g = if scale != 1.0 {
                bmul(&g, &step_scalar(scale, dev))?
            } else {
                g
            };
            // Fused in-place step: parameter and velocity keep their storage.
            let stepped = if self.momentum == 0.0 {
                raw_axpy_("sgd_step", -self.lr, &cur, &g).is_ok()
            } else {
                let v = self.velocity[i].get_or_insert_with(|| zero_like(&cur));
                raw_sgd_step_(&cur, v, &g, self.lr, self.momentum, self.nesterov).is_ok()
            };
            if stepped {
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
        }
        Ok(())
    }
}

impl OptimizerState for Sgd {
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        // Cold path: state comes back to the host for serialization.
        self.velocity
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let host = v
                    .as_ref()
                    .map(|t| t.to_device(Device::Cpu).expect("cpu is always registered"))
                    .unwrap_or_else(|| zero_like_cpu(self.params[i].tensor().numel()));
                // State is stored flat (one element per parameter), matching
                // the pre-device file format.
                let n = host.numel();
                (
                    format!("velocity.{i}"),
                    host.reshape(&[n]).expect("flat reshape"),
                )
            })
            .collect()
    }

    fn restore(&mut self, tensors: &[(String, Tensor)]) -> Result<()> {
        if tensors.len() != self.velocity.len() {
            return Err(Error::Format {
                op: "optim_restore",
                msg: format!(
                    "expected {} velocity buffers, got {}",
                    self.velocity.len(),
                    tensors.len()
                ),
            });
        }
        for i in 0..self.velocity.len() {
            check_buf(
                &mut self.velocity[i],
                self.params[i].tensor().shape(),
                &format!("velocity.{i}"),
                tensors,
            )?;
        }
        Ok(())
    }
}

fn zero_like_cpu(len: usize) -> Tensor {
    Tensor::from_vec(vec![0.0; len], &[len]).expect("flat f32 buffer")
}

/// Adam with bias correction. Defaults: beta1=0.9, beta2=0.999, eps=1e-8.
pub struct Adam {
    params: Vec<Param>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    max_grad_norm: Option<f32>,
    /// Per-parameter step counts: a param that skips a step (no grad) must not
    /// advance its bias correction, or its next update is scaled wrongly.
    t: Vec<u32>,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
}

impl Adam {
    pub fn new(params: Vec<Param>, lr: f32) -> Adam {
        let m = params.iter().map(|_| None).collect();
        let v = params.iter().map(|_| None).collect();
        let t = vec![0u32; params.len()];
        Adam {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            max_grad_norm: None,
            t,
            m,
            v,
        }
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Adam {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_eps(mut self, eps: f32) -> Adam {
        self.eps = eps;
        self
    }

    pub fn with_max_grad_norm(mut self, max_norm: f32) -> Adam {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// The learning rate the next step will use.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set the learning rate for subsequent steps (`LrScheduler` seam).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
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
        let scale = clip_scale(&self.params, self.max_grad_norm);
        for i in 0..self.params.len() {
            let Some(g) = self.params[i].grad() else {
                continue;
            };
            let p = &self.params[i];
            let cur = p.tensor();
            let dev = cur.device();
            self.t[i] += 1;
            let bc1 = 1.0 - self.beta1.powi(self.t[i] as i32);
            let bc2 = 1.0 - self.beta2.powi(self.t[i] as i32);
            let g = if scale != 1.0 {
                bmul(&g, &step_scalar(scale, dev))?
            } else {
                g
            };
            let m = self.m[i].get_or_insert_with(|| zero_like(&cur));
            let v = self.v[i].get_or_insert_with(|| zero_like(&cur));
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
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        // Cold path: state comes back to the host for serialization.
        let mut out = Vec::new();
        for i in 0..self.params.len() {
            let host = |s: &Option<Tensor>| -> Tensor {
                let t = s
                    .as_ref()
                    .map(|t| t.to_device(Device::Cpu).expect("cpu is always registered"))
                    .unwrap_or_else(|| zero_like_cpu(self.params[i].tensor().numel()));
                // Stored flat (one element per parameter): file-format parity.
                let n = t.numel();
                t.reshape(&[n]).expect("flat reshape")
            };
            out.push((format!("m.{i}"), host(&self.m[i])));
            out.push((format!("v.{i}"), host(&self.v[i])));
            out.push((format!("t.{i}"), Tensor::scalar(self.t[i] as f32)));
        }
        out
    }

    fn restore(&mut self, tensors: &[(String, Tensor)]) -> Result<()> {
        if tensors.len() != 3 * self.params.len() {
            return Err(Error::Format {
                op: "optim_restore",
                msg: format!(
                    "expected {} arrays, got {}",
                    3 * self.params.len(),
                    tensors.len()
                ),
            });
        }
        for i in 0..self.params.len() {
            let shape = self.params[i].tensor().shape().to_vec();
            check_buf(&mut self.m[i], &shape, &format!("m.{i}"), tensors)?;
            check_buf(&mut self.v[i], &shape, &format!("v.{i}"), tensors)?;
            let ts = tensors
                .iter()
                .find(|(n, _)| n == &format!("t.{i}"))
                .ok_or_else(|| Error::Format {
                    op: "optim_restore",
                    msg: format!("optimizer state is missing t.{i}"),
                })?;
            let tv = ts.1.to_vec()[0];
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
pub struct AdamW {
    params: Vec<Param>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    max_grad_norm: Option<f32>,
    t: u32,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    /// When set, the step runs the CUDA-graph-capturable path: the timestep
    /// lives in a device tensor `[step, bc1, bc2]` and bias correction advances
    /// in-kernel, so a captured step replays correctly (mirrors PyTorch
    /// `capturable=True`). `None` = the ordinary host-timestep path.
    capturable_t: Option<Tensor>,
}

impl AdamW {
    pub fn new(params: Vec<Param>, lr: f32) -> AdamW {
        let m = params.iter().map(|_| None).collect();
        let v = params.iter().map(|_| None).collect();
        AdamW {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            max_grad_norm: None,
            t: 0,
            m,
            v,
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
    /// CUDA device (the capturable step calls device-only primitives). On CPU
    /// params, mixed-device sets, or an empty param list this is a no-op and the
    /// optimiser keeps the ordinary host path — `is_capturable()` then stays
    /// `false`. `lr` and betas are fixed for the life of a captured graph
    /// (re-capture to change them).
    pub fn capturable(mut self) -> AdamW {
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
        // Seed the device timestep from the CURRENT host counter so enabling
        // capture AFTER eager warm-up steps preserves the timestep (and thus the
        // bias correction that matches the already-matured moment buffers).
        // `step` starts at self.t; if that's 0 the first increment yields t=1.
        let t0 = self.t as f32;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
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
        self
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> AdamW {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_max_grad_norm(mut self, max_norm: f32) -> AdamW {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// The learning rate the next step will use.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set the learning rate for subsequent steps (`LrScheduler` seam).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
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
        if let Some(t_dev) = self.capturable_t.clone() {
            return self.update_capturable(&t_dev);
        }
        let scale = clip_scale(&self.params, self.max_grad_norm);
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..self.params.len() {
            let Some(g) = self.params[i].grad() else {
                continue;
            };
            let p = &self.params[i];
            let cur = p.tensor();
            let dev = cur.device();
            let g = if scale != 1.0 {
                bmul(&g, &step_scalar(scale, dev))?
            } else {
                g
            };
            let m = self.m[i].get_or_insert_with(|| zero_like(&cur));
            let v = self.v[i].get_or_insert_with(|| zero_like(&cur));
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
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        // Cold path: state comes back to the host for serialization.
        // In capturable mode the device timestep buffer is authoritative
        // (the host `self.t` is not advanced by replays), so read `step` back
        // from it; otherwise the host counter is the source of truth.
        let step = match &self.capturable_t {
            Some(t) => t.to_device(Device::Cpu).expect("cpu is always registered").to_vec()[0] as u32,
            None => self.t,
        };
        let mut out = vec![("t".to_string(), Tensor::scalar(step as f32))];
        let host = |s: &Option<Tensor>| -> Tensor {
            let t = s
                .as_ref()
                .map(|t| t.to_device(Device::Cpu).expect("cpu is always registered"))
                .unwrap_or_else(|| zero_like_cpu(self.params[0].tensor().numel()));
            // Stored flat (one element per parameter): file-format parity.
            let n = t.numel();
            t.reshape(&[n]).expect("flat reshape")
        };
        for i in 0..self.params.len() {
            out.push((format!("m.{i}"), host(&self.m[i])));
            out.push((format!("v.{i}"), host(&self.v[i])));
        }
        out
    }

    fn restore(&mut self, tensors: &[(String, Tensor)]) -> Result<()> {
        if tensors.len() != 2 * self.params.len() + 1 {
            return Err(Error::Format {
                op: "optim_restore",
                msg: format!(
                    "expected {} arrays, got {}",
                    2 * self.params.len() + 1,
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
        let tv = ts.1.to_vec()[0];
        if tv < 0.0 || tv.fract() != 0.0 || tv > u32::MAX as f32 {
            return Err(Error::Format {
                op: "optim_restore",
                msg: format!("t must be a non-negative integer, got {tv}"),
            });
        }
        self.t = tv as u32;
        // In capturable mode the device buffer is authoritative, so push the
        // restored timestep + its bias correction back onto the device;
        // otherwise a resumed run would replay from step 0 (frozen correction).
        if self.capturable_t.is_some() {
            let bc1 = 1.0 - self.beta1.powi(self.t as i32);
            let bc2 = 1.0 - self.beta2.powi(self.t as i32);
            let dev = self.params[0].tensor().device();
            let restored =
                Tensor::from_vec(vec![self.t as f32, bc1, bc2], &[3]).and_then(|t| t.to_device(dev))?;
            self.capturable_t = Some(restored);
        }
        for i in 0..self.params.len() {
            let shape = self.params[i].tensor().shape().to_vec();
            check_buf(&mut self.m[i], &shape, &format!("m.{i}"), tensors)?;
            check_buf(&mut self.v[i], &shape, &format!("v.{i}"), tensors)?;
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
