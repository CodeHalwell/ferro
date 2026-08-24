//! Additional nn layers (Conv2D, BatchNorm, Dropout) and the ModuleList
//! container, extending `crate::nn`. All forwards compose recorded autograd
//! ops, so gradients flow without custom backwards.

use std::cell::{Cell, RefCell};

use crate::error::{Error, Result};
use crate::nn::{Init, Module};
use crate::params::Param;
use crate::rng::Rng;
use crate::tensor::Tensor;

/// Ordered module container with torch-style indexed parameter names. Unlike
/// Sequential it also exposes its layers for direct use.
pub struct ModuleList {
    pub layers: Vec<Box<dyn Module>>,
}

impl ModuleList {
    pub fn new(layers: Vec<Box<dyn Module>>) -> ModuleList {
        ModuleList { layers }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Module for ModuleList {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, l)| {
                l.named_parameters()
                    .into_iter()
                    .map(move |(n, p)| (format!("{i}.{n}"), p))
            })
            .collect()
    }

    fn set_training(&self, training: bool) {
        for l in &self.layers {
            l.set_training(training);
        }
    }
}

/// 2-D convolution over NCHW input with a `[c_out]` bias, wrapping the
/// im2col+GEMM op from `ops_ext::conv2d` (stride + zero padding, no
/// dilation/groups). Weights are kaiming-normal initialized.
pub struct Conv2D {
    weight: Param,
    bias: Param,
    stride: usize,
    padding: usize,
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel: usize, rng: &Rng) -> Conv2D {
        Conv2D::with_config(in_channels, out_channels, kernel, 1, 0, rng)
    }

    pub fn with_config(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        rng: &Rng,
    ) -> Conv2D {
        let fan_in = in_channels * kernel * kernel;
        let w = Init::Kaiming.fill(
            rng,
            &[out_channels, in_channels, kernel, kernel],
            fan_in,
            fan_in,
        );
        Conv2D {
            weight: Param::new(w),
            bias: Param::new(Tensor::zeros(&[out_channels])),
            stride,
            padding,
        }
    }
}

impl Module for Conv2D {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 4 {
            return Err(Error::InvalidShape {
                op: "conv2d",
                msg: format!("input must be 4-D NCHW, got {:?}", x.shape()),
            });
        }
        let y = x.conv2d(&self.weight.tensor(), self.stride, self.padding)?;
        // Reshape [c_out] to [1, c_out, 1, 1] so it broadcasts along the
        // channel axis; a bare [c_out] would align with W instead.
        let c_out = self.bias.tensor().shape()[0];
        let b = self.bias.tensor().reshape(&[1, c_out, 1, 1])?;
        y.add(&b)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![
            ("weight".into(), self.weight.clone()),
            ("bias".into(), self.bias.clone()),
        ]
    }
}

/// Batch normalization over the batch dim of a `[batch, features]` input.
/// Training normalizes with batch statistics and updates exponential running
/// stats (momentum 0.1, unbiased running variance like torch); evaluation
/// normalizes with the frozen running stats. Composed from autograd ops, so
/// gamma/beta and the input all receive gradients in train mode.
pub struct BatchNorm {
    gamma: Param,
    beta: Param,
    eps: f32,
    training: Cell<bool>,
    running_mean: RefCell<Vec<f32>>,
    running_var: RefCell<Vec<f32>>,
}

impl BatchNorm {
    pub fn new(features: usize) -> BatchNorm {
        BatchNorm {
            gamma: Param::new(Tensor::ones(&[features])),
            beta: Param::new(Tensor::zeros(&[features])),
            eps: 1e-5,
            training: Cell::new(true),
            running_mean: RefCell::new(vec![0.0; features]),
            running_var: RefCell::new(vec![1.0; features]),
        }
    }
}

impl Module for BatchNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 2 {
            return Err(Error::InvalidShape {
                op: "batch_norm",
                msg: format!("input must be 2-D [batch, features], got {:?}", x.shape()),
            });
        }
        let f = x.shape()[1];
        if self.gamma.tensor().numel() != f {
            return Err(Error::ShapeMismatch {
                op: "batch_norm",
                lhs: x.shape().to_vec(),
                rhs: vec![f],
            });
        }
        let norm = if self.training.get() {
            let mu = x.mean_dim(0, true)?;
            let centered = x.sub(&mu)?;
            // Biased variance for normalization...
            let var = centered.mul(&centered)?.mean_dim(0, true)?;
            // ...unbiased for the running estimate (torch semantics).
            let n = x.shape()[0] as f32;
            let unbiased_scale = n / (n - 1.0).max(1.0);
            let mu_v = mu.to_vec();
            let var_v = var.to_vec();
            let mut rm = self.running_mean.borrow_mut();
            let mut rv = self.running_var.borrow_mut();
            for j in 0..f {
                rm[j] = 0.9 * rm[j] + 0.1 * mu_v[j];
                rv[j] = 0.9 * rv[j] + 0.1 * var_v[j] * unbiased_scale;
            }
            centered.div(&var.add(&Tensor::scalar(self.eps))?.sqrt())?
        } else {
            let rm = Tensor::from_vec(self.running_mean.borrow().clone(), &[f])?;
            let rv = Tensor::from_vec(self.running_var.borrow().clone(), &[f])?;
            x.sub(&rm)?
                .div(&rv.add(&Tensor::scalar(self.eps))?.sqrt())?
        };
        norm.mul(&self.gamma.tensor())?.add(&self.beta.tensor())
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![
            ("weight".into(), self.gamma.clone()),
            ("bias".into(), self.beta.clone()),
        ]
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

/// Inverted dropout backed by the counter-based Philox op: in training mode
/// each activation is zeroed with probability p and survivors scaled by
/// 1/(1-p); evaluation is the identity. Each training forward advances an
/// internal stream offset so every step samples a fresh mask, while the
/// sequence stays deterministic given (seed, forward count).
pub struct Dropout {
    p: f32,
    seed: u64,
    training: Cell<bool>,
    offset: Cell<u64>,
}

impl Dropout {
    pub fn new(p: f32) -> Dropout {
        Dropout {
            p,
            seed: 0,
            training: Cell::new(true),
            offset: Cell::new(0),
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Dropout {
        self.seed = seed;
        self
    }

    /// Stream position of the next training mask; save it in checkpoints to
    /// resume sampling exactly where training stopped.
    pub fn rng_offset(&self) -> u64 {
        self.offset.get()
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !self.training.get() {
            return x.dropout(self.p, false, self.seed, 0);
        }
        let y = x.dropout(self.p, true, self.seed, self.offset.get());
        // Advance by the element count so the next forward draws a fresh,
        // non-overlapping slice of the Philox stream.
        self.offset.set(self.offset.get() + x.numel() as u64);
        y
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        Vec::new()
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}
