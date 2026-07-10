//! Neural-network building blocks (Linear, activations, sequential containers).
//!
//! Built on the frozen `Tensor` API from `crate::tensor` / `crate::ops`:
//! `matmul`, `add` (broadcasts a bias row), `relu`, `sigmoid`, and the autograd
//! `backward()`.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::params::Param;
use crate::rng::Rng;
use crate::tensor::Tensor;

pub trait Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<Param>;
}

/// Affine layer `y = x @ W + b` with He-initialized weights.
pub struct Linear {
    weight: Param,
    bias: Param,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, rng: &Rng) -> Linear {
        let scale = (2.0 / in_features as f32).sqrt();
        let w: Vec<f32> = (0..in_features * out_features).map(|_| rng.normal() * scale).collect();
        let weight = Param::new(Tensor::from_vec(w, &[in_features, out_features]).unwrap());
        let bias = Param::new(Tensor::zeros(&[out_features]));
        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight.tensor())?.add(&self.bias.tensor())
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

pub struct Relu;

impl Module for Relu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.relu())
    }

    fn parameters(&self) -> Vec<Param> {
        Vec::new()
    }
}

pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.sigmoid())
    }

    fn parameters(&self) -> Vec<Param> {
        Vec::new()
    }
}

/// Layer normalization over the last dim of a `[batch, dim]` input (2-D only
/// for now): `(x - mean) / sqrt(var + eps) * gamma + beta` with learnable
/// per-feature `gamma`/`beta`. Composed from autograd ops, so the gradient
/// flows without a custom backward.
pub struct LayerNorm {
    gamma: Param,
    beta: Param,
    eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize) -> LayerNorm {
        let gamma = Param::new(Tensor::ones(&[dim]));
        let beta = Param::new(Tensor::zeros(&[dim]));
        LayerNorm { gamma, beta, eps: 1e-5 }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Reducing dim 1 while gamma/beta broadcast over the last dim only
        // agrees when both are the feature dim, i.e. for 2-D input.
        if x.ndim() != 2 {
            return Err(Error::InvalidShape {
                op: "layer_norm",
                msg: format!("input must be 2-D [batch, dim], got {:?}", x.shape()),
            });
        }
        let mu = x.mean_dim(1, true)?;
        let centered = x.sub(&mu)?;
        let var = centered.mul(&centered)?.mean_dim(1, true)?;
        let norm = centered.div(&var.add(&Tensor::scalar(self.eps))?.sqrt())?;
        norm.mul(&self.gamma.tensor())?.add(&self.beta.tensor())
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

/// Runs its layers in order, threading the output of each into the next.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    fn parameters(&self) -> Vec<Param> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

/// Cross-entropy loss for `[batch, classes]` logits against one-hot targets of
/// the same shape: mean over the batch of -sum(target * log_softmax(logits)).
/// Composed from autograd ops, so the gradient flows without a custom backward.
pub fn cross_entropy(logits: &Tensor, targets_one_hot: &Tensor) -> Result<Tensor> {
    // Exact shape match: mul broadcasts, so a [1, classes] or [classes] target
    // against a batch would silently train every row on the same label.
    if targets_one_hot.shape() != logits.shape() {
        return Err(Error::ShapeMismatch {
            op: "cross_entropy",
            lhs: logits.shape().to_vec(),
            rhs: targets_one_hot.shape().to_vec(),
        });
    }
    let lp = logits.log_softmax(1)?;
    // log_softmax may fall back to the host for device logits, so realign
    // constant targets to its device. Targets that require grad cannot be
    // moved silently (to_device detaches), so those keep the strict path.
    let targets = if targets_one_hot.device() != lp.device() && !targets_one_hot.requires_grad() {
        targets_one_hot.to_device(lp.device())?
    } else {
        targets_one_hot.clone()
    };
    Ok(lp.mul(&targets)?.sum_dim(1, false)?.neg().mean())
}

/// One-hot encode 1-D I64 class ids `[n]` into an f32 `[n, classes]` tensor.
pub fn one_hot(ids: &Tensor, classes: usize) -> Result<Tensor> {
    if ids.dtype() != DType::I64 {
        return Err(Error::DtypeMismatch { op: "one_hot", expected: DType::I64, got: ids.dtype() });
    }
    if ids.ndim() != 1 {
        return Err(Error::InvalidShape {
            op: "one_hot",
            msg: format!("ids must be 1-D, got shape {:?}", ids.shape()),
        });
    }
    let idx = ids.to_vec_i64();
    let mut data = vec![0f32; idx.len() * classes];
    for (row, &id) in idx.iter().enumerate() {
        if id < 0 || id as usize >= classes {
            return Err(Error::InvalidShape {
                op: "one_hot",
                msg: format!("id {id} out of range for {classes} classes"),
            });
        }
        data[row * classes + id as usize] = 1.0;
    }
    Tensor::from_vec(data, &[idx.len(), classes])
}

/// Cross-entropy for `[batch, classes]` logits against I64 class ids `[batch]`
/// (like PyTorch's `F.cross_entropy` with integer targets): one-hot encode the
/// ids, then reuse `cross_entropy`.
pub fn cross_entropy_indices(logits: &Tensor, target_ids: &Tensor) -> Result<Tensor> {
    if logits.ndim() != 2 {
        return Err(Error::InvalidShape {
            op: "cross_entropy_indices",
            msg: format!("logits must be 2-D [batch, classes], got {:?}", logits.shape()),
        });
    }
    if target_ids.numel() != logits.shape()[0] {
        return Err(Error::InvalidShape {
            op: "cross_entropy_indices",
            msg: format!("{} target ids for batch of {}", target_ids.numel(), logits.shape()[0]),
        });
    }
    cross_entropy(logits, &one_hot(target_ids, logits.shape()[1])?)
}
