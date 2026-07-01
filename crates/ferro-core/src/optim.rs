//! Optimizers (SGD, Adam) operating on parameter tensors and their `.grad()`.
//!
//! Tensors are immutable and `Arc`-shared, so a step never mutates in place:
//! it reads `param.tensor()` and `param.grad()` as `Vec<f32>`, computes the new
//! leaf values, and re-installs them via `Param::set`. Optimizer state (momentum
//! buffers, Adam moments, timestep) lives here as plain `Vec<f32>`, one entry
//! per parameter element.

use crate::params::Param;
use crate::tensor::Tensor;
use crate::Result;

/// Stochastic gradient descent with optional (heavy-ball) momentum.
pub struct Sgd {
    params: Vec<Param>,
    lr: f32,
    momentum: f32,
    velocity: Vec<Vec<f32>>,
}

impl Sgd {
    pub fn new(params: Vec<Param>, lr: f32) -> Sgd {
        let velocity = params.iter().map(|p| vec![0.0; p.tensor().numel()]).collect();
        Sgd { params, lr, momentum: 0.0, velocity }
    }

    pub fn with_momentum(mut self, m: f32) -> Sgd {
        self.momentum = m;
        self
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        for (i, p) in self.params.iter().enumerate() {
            let grad = match p.grad() {
                Some(g) => g.to_vec(),
                None => continue,
            };
            let cur = p.tensor();
            let mut vals = cur.to_vec();
            let v = &mut self.velocity[i];
            for j in 0..vals.len() {
                v[j] = self.momentum * v[j] + grad[j];
                vals[j] -= self.lr * v[j];
            }
            set_leaf(p, vals, cur.shape());
        }
    }
}

/// Adam with bias correction. Defaults: beta1=0.9, beta2=0.999, eps=1e-8.
pub struct Adam {
    params: Vec<Param>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: u32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Adam {
    pub fn new(params: Vec<Param>, lr: f32) -> Adam {
        let m = params.iter().map(|p| vec![0.0; p.tensor().numel()]).collect();
        let v = params.iter().map(|p| vec![0.0; p.tensor().numel()]).collect();
        Adam { params, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m, v }
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

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&mut self) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (i, p) in self.params.iter().enumerate() {
            let grad = match p.grad() {
                Some(g) => g.to_vec(),
                None => continue,
            };
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

fn set_leaf(p: &Param, vals: Vec<f32>, shape: &[usize]) {
    let updated: Result<Tensor> = Tensor::from_vec(vals, shape);
    p.set(updated.expect("optimizer rebuilds a leaf with the same shape"));
}
