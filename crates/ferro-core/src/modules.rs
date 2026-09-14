//! Additional nn layers (Conv2D, BatchNorm, Dropout) and the ModuleList
//! container, extending `crate::nn`. Forwards use the existing recorded ops;
//! BatchNorm uses the functional rank-2/rank-4 first-order implementation.

use std::cell::Cell;

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

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        self.layers.iter().enumerate().flat_map(|(i, l)| l.named_buffers().into_iter().map(move |(n, t)| (format!("{i}.{n}"), t))).collect()
    }
    fn snapshot_scalars(&self) -> Vec<(String, u64)> {
        self.layers.iter().enumerate().flat_map(|(i, l)| l.snapshot_scalars().into_iter().map(move |(n, v)| (format!("{i}.{n}"), v))).collect()
    }
    fn validate_scalars(&self, state: &[(String, u64)]) -> Result<()> {
        crate::nn::validate_scalar_keys(&self.snapshot_scalars(), state)?;
        for (i, l) in self.layers.iter().enumerate() { l.validate_scalars(&crate::nn::child_scalars(state, i))?; }
        Ok(())
    }
    fn commit_scalars(&self, state: &[(String, u64)]) {
        for (i, l) in self.layers.iter().enumerate() { l.commit_scalars(&crate::nn::child_scalars(state, i)); }
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

/// Batch normalization for `[N, C]` or NCHW `[N, C, H, W]` inputs.
/// Training normalizes with batch statistics and updates exponential running
/// stats (momentum 0.1, unbiased running variance like torch); evaluation
/// normalizes with frozen running stats and accepts singleton/empty batches.
/// The functional op computes on CPU; no resident-device support is claimed.
/// Buffers remain live, stable CPU f32 tensors across updates and restores.
pub struct BatchNorm {
    gamma: Param,
    beta: Param,
    eps: f32,
    training: Cell<bool>,
    running_mean: Tensor,
    running_var: Tensor,
}

impl BatchNorm {
    pub fn new(features: usize) -> BatchNorm {
        BatchNorm {
            gamma: Param::new(Tensor::ones(&[features])),
            beta: Param::new(Tensor::zeros(&[features])),
            eps: 1e-5,
            training: Cell::new(true),
            running_mean: Tensor::zeros(&[features]),
            running_var: Tensor::ones(&[features]),
        }
    }
}

impl Module for BatchNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let result = x.batch_norm(&self.gamma.tensor(), &self.beta.tensor(),
            &self.running_mean, &self.running_var, self.eps, self.training.get(), 0.1)?;
        if self.training.get() {
            crate::inplace::raw_copy_("batch_norm", &self.running_mean, &result.running_mean)?;
            crate::inplace::raw_copy_("batch_norm", &self.running_var, &result.running_var)?;
        }
        Ok(result.output)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![
            ("weight".into(), self.gamma.clone()),
            ("bias".into(), self.beta.clone()),
        ]
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        vec![("running_mean".into(), self.running_mean.clone()), ("running_var".into(), self.running_var.clone())]
    }
    fn snapshot_scalars(&self) -> Vec<(String, u64)> { vec![("training".into(), self.training.get() as u64)] }
    fn validate_scalars(&self, state: &[(String, u64)]) -> Result<()> {
        crate::nn::validate_scalar_keys(&self.snapshot_scalars(), state)?;
        if crate::nn::scalar(state, "training") > 1 { return Err(Error::Format { op: "module_state", msg: "invalid training mode".into() }); }
        Ok(())
    }
    fn commit_scalars(&self, state: &[(String, u64)]) { self.training.set(crate::nn::scalar(state, "training") != 0); }
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
    seed: Cell<u64>,
    training: Cell<bool>,
    offset: Cell<u64>,
}

impl Dropout {
    pub fn new(p: f32) -> Dropout {
        Dropout {
            p,
            seed: Cell::new(0),
            training: Cell::new(true),
            offset: Cell::new(0),
        }
    }

    pub fn with_seed(self, seed: u64) -> Dropout {
        self.seed.set(seed);
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
            return x.dropout(self.p, false, self.seed.get(), 0);
        }
        let next = self.offset.get().checked_add(x.numel() as u64)
            .ok_or_else(|| Error::Format { op: "dropout", msg: "RNG counter overflow".into() })?;
        let y = x.dropout(self.p, true, self.seed.get(), self.offset.get())?;
        self.offset.set(next);
        Ok(y)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        Vec::new()
    }

    fn snapshot_scalars(&self) -> Vec<(String, u64)> {
        vec![("training".into(), self.training.get() as u64), ("seed".into(), self.seed.get()),
             ("offset".into(), self.offset.get()), ("p".into(), self.p.to_bits() as u64)]
    }
    fn validate_scalars(&self, state: &[(String, u64)]) -> Result<()> {
        crate::nn::validate_scalar_keys(&self.snapshot_scalars(), state)?;
        if crate::nn::scalar(state, "training") > 1 || crate::nn::scalar(state, "p") != self.p.to_bits() as u64 {
            return Err(Error::Format { op: "module_state", msg: "invalid dropout mode or probability mismatch".into() });
        }
        Ok(())
    }
    fn commit_scalars(&self, state: &[(String, u64)]) {
        self.training.set(crate::nn::scalar(state, "training") != 0);
        self.seed.set(crate::nn::scalar(state, "seed"));
        self.offset.set(crate::nn::scalar(state, "offset"));
    }

    fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}
