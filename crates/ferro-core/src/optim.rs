//! Optimizers (SGD, Adam, AdamW), LR schedulers, and gradient clipping,
//! operating on parameter tensors and their `.grad()`.
//!
//! Tensors are immutable and `Arc`-shared, so a step never mutates in place:
//! it reads `param.tensor()` and `param.grad()` as `Vec<f32>`, computes the new
//! leaf values, and re-installs them via `Param::set`. Optimizer state (momentum
//! buffers, Adam moments, timestep) lives here as plain `Vec<f32>`, one entry
//! per parameter element.

use crate::error::Error;
use crate::params::Param;
use crate::tensor::Tensor;
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

fn state_tensor(vals: Vec<f32>) -> Tensor {
    let len = vals.len();
    Tensor::from_vec(vals, &[len]).expect("flat f32 buffer")
}

fn check_buf(buf: &mut Vec<f32>, key: &str, tensors: &[(String, Tensor)]) -> Result<()> {
    let t = tensors
        .iter()
        .find(|(n, _)| n == key)
        .ok_or_else(|| Error::Format {
            op: "optim_restore",
            msg: format!("optimizer state is missing {key:?}"),
        })?;
    if t.1.shape() != [buf.len()] || t.1.dtype() != crate::dtype::DType::F32 {
        return Err(Error::Format {
            op: "optim_restore",
            msg: format!(
                "{key:?}: expected f32 [{}], got {:?} {:?}",
                buf.len(),
                t.1.dtype(),
                t.1.shape()
            ),
        });
    }
    *buf = t.1.to_vec();
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
    velocity: Vec<Vec<f32>>,
}

impl Sgd {
    pub fn new(params: Vec<Param>, lr: f32) -> Sgd {
        let velocity = params
            .iter()
            .map(|p| vec![0.0; p.tensor().numel()])
            .collect();
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

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        let scale = clip_scale(&self.params, self.max_grad_norm);
        for (i, p) in self.params.iter().enumerate() {
            let grad = match grads(&self.params[i], scale) {
                Some(g) => g,
                None => continue,
            };
            let cur = p.tensor();
            let mut vals = cur.to_vec();
            let v = &mut self.velocity[i];
            for j in 0..vals.len() {
                v[j] = self.momentum * v[j] + grad[j];
                let d = if self.nesterov {
                    self.momentum * v[j] + grad[j]
                } else {
                    v[j]
                };
                vals[j] -= self.lr * d;
            }
            set_leaf(p, vals, cur.shape());
        }
    }
}

impl OptimizerState for Sgd {
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        self.velocity
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("velocity.{i}"), state_tensor(v.clone())))
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
            check_buf(&mut self.velocity[i], &format!("velocity.{i}"), tensors)?;
        }
        Ok(())
    }
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
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Adam {
    pub fn new(params: Vec<Param>, lr: f32) -> Adam {
        let m = params
            .iter()
            .map(|p| vec![0.0; p.tensor().numel()])
            .collect();
        let v = params
            .iter()
            .map(|p| vec![0.0; p.tensor().numel()])
            .collect();
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

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        let scale = clip_scale(&self.params, self.max_grad_norm);
        for i in 0..self.params.len() {
            let grad = match grads(&self.params[i], scale) {
                Some(g) => g,
                None => continue,
            };
            let p = &self.params[i];
            self.t[i] += 1;
            let bc1 = 1.0 - self.beta1.powi(self.t[i] as i32);
            let bc2 = 1.0 - self.beta2.powi(self.t[i] as i32);
            let cur = p.tensor();
            let mut vals = cur.to_vec();
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            for j in 0..vals.len() {
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * grad[j];
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * grad[j] * grad[j];
                let m_hat = m[j] / bc1;
                let v_hat = v[j] / bc2;
                vals[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            set_leaf(p, vals, cur.shape());
        }
    }
}

impl OptimizerState for Adam {
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for i in 0..self.params.len() {
            out.push((format!("m.{i}"), state_tensor(self.m[i].clone())));
            out.push((format!("v.{i}"), state_tensor(self.v[i].clone())));
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
            check_buf(&mut self.m[i], &format!("m.{i}"), tensors)?;
            check_buf(&mut self.v[i], &format!("v.{i}"), tensors)?;
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
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(params: Vec<Param>, lr: f32) -> AdamW {
        let m = params
            .iter()
            .map(|p| vec![0.0; p.tensor().numel()])
            .collect();
        let v = params
            .iter()
            .map(|p| vec![0.0; p.tensor().numel()])
            .collect();
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
        }
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

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        let scale = clip_scale(&self.params, self.max_grad_norm);
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..self.params.len() {
            let grad = match grads(&self.params[i], scale) {
                Some(g) => g,
                None => continue,
            };
            let p = &self.params[i];
            let cur = p.tensor();
            let mut vals = cur.to_vec();
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            for j in 0..vals.len() {
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * grad[j];
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * grad[j] * grad[j];
                let m_hat = m[j] / bc1;
                let v_hat = v[j] / bc2;
                vals[j] -=
                    self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * vals[j]);
            }
            set_leaf(p, vals, cur.shape());
        }
    }
}

impl OptimizerState for AdamW {
    fn snapshot(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![("t".to_string(), Tensor::scalar(self.t as f32))];
        for i in 0..self.params.len() {
            out.push((format!("m.{i}"), state_tensor(self.m[i].clone())));
            out.push((format!("v.{i}"), state_tensor(self.v[i].clone())));
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
        for i in 0..self.params.len() {
            check_buf(&mut self.m[i], &format!("m.{i}"), tensors)?;
            check_buf(&mut self.v[i], &format!("v.{i}"), tensors)?;
        }
        Ok(())
    }
}

// --- gradient clipping ----------------------------------------------------

/// Global L2 norm over every parameter's accumulated gradient (params without
/// a grad contribute nothing).
pub fn global_grad_norm(params: &[Param]) -> f32 {
    params
        .iter()
        .filter_map(|p| p.grad())
        .map(|g| g.to_vec().iter().map(|&x| x * x).sum::<f32>())
        .sum::<f32>()
        .sqrt()
}

/// The multiplicative factor applied when clipping to `max` (None or already
/// under the budget gives 1.0).
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

fn grads(p: &Param, scale: f32) -> Option<Vec<f32>> {
    p.grad()
        .map(|g| g.to_vec().into_iter().map(|x| x * scale).collect())
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

fn set_leaf(p: &Param, vals: Vec<f32>, shape: &[usize]) {
    // Step math runs on host Vecs, but the leaf must go back to wherever the
    // parameter lives or a device param would silently migrate to cpu.
    let device = p.tensor().device();
    let updated: Result<Tensor> = Tensor::from_vec(vals, shape);
    let host = updated.expect("optimizer rebuilds a leaf with the same shape");
    p.set(
        host.to_device(device)
            .expect("param's device backend is registered"),
    );
}
