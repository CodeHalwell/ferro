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

pub struct Gelu;

impl Module for Gelu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.gelu())
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

/// RMS normalization over the last dim (any rank, unlike the 2-D LayerNorm):
/// `x / sqrt(mean(x^2) + eps) * gamma` with learnable per-feature `gamma`.
/// Composed from autograd ops, so the gradient flows without a custom backward.
pub struct RmsNorm {
    gamma: Param,
    eps: f32,
}

impl RmsNorm {
    pub fn new(dim: usize) -> RmsNorm {
        RmsNorm { gamma: Param::new(Tensor::ones(&[dim])), eps: 1e-5 }
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last = x.ndim() - 1;
        let ms = x.mul(x)?.mean_dim(last, true)?;
        x.div(&ms.add(&Tensor::scalar(self.eps))?.sqrt())?.mul(&self.gamma.tensor())
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.gamma.clone()]
    }
}

/// Token embedding: a `[num_embeddings, dim]` N(0,1) weight looked up by I64
/// ids of any shape; the output appends `dim` (ids `[b, s]` -> `[b, s, dim]`).
/// Lookup goes through the recorded `embedding` op, so duplicate ids
/// scatter-add their grads into the weight.
pub struct Embedding {
    weight: Param,
    dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, dim: usize, rng: &Rng) -> Embedding {
        let w: Vec<f32> = (0..num_embeddings * dim).map(|_| rng.normal()).collect();
        Embedding { weight: Param::new(Tensor::from_vec(w, &[num_embeddings, dim]).unwrap()), dim }
    }
}

impl Module for Embedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let flat = if ids.ndim() == 1 { ids.clone() } else { ids.reshape(&[ids.numel()])? };
        let out = crate::ops_ext::embedding(&self.weight.tensor(), &flat)?;
        let mut shape = ids.shape().to_vec();
        shape.push(self.dim);
        out.reshape(&shape)
    }

    fn parameters(&self) -> Vec<Param> {
        vec![self.weight.clone()]
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
    Ok(logits.log_softmax(1)?.mul(targets_one_hot)?.sum_dim(1, false)?.neg().mean())
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

/// Scaled dot-product attention over `[batch, seq, head_dim]` inputs (fold
/// heads into batch before calling): softmax(q k^T / sqrt(d) + mask) v, with
/// an optional causal mask (position i attends to j <= i; masked scores get
/// -1e9 rather than -inf so softmax stays NaN-free). Composed from autograd
/// ops, so gradients flow to q, k, and v without a custom backward.
pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.ndim() != 3 {
            return Err(Error::InvalidShape {
                op: "scaled_dot_product_attention",
                msg: format!("{name} must be 3-D [batch, seq, head_dim], got {:?}", t.shape()),
            });
        }
    }
    let (b, sq, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let sk = k.shape()[1];
    if k.shape() != [b, sk, d] || v.shape()[0] != b || v.shape()[1] != sk {
        return Err(Error::InvalidShape {
            op: "scaled_dot_product_attention",
            msg: format!("incompatible shapes q {:?}, k {:?}, v {:?}", q.shape(), k.shape(), v.shape()),
        });
    }
    let scale = Tensor::scalar(1.0 / (d as f32).sqrt());
    let mut scores = q.bmm(&k.transpose(1, 2)?)?.mul(&scale)?;
    if causal {
        let mut m = vec![0.0f32; sq * sk];
        for i in 0..sq {
            for j in 0..sk {
                if j > i {
                    m[i * sk + j] = -1e9;
                }
            }
        }
        scores = scores.add(&Tensor::from_vec(m, &[sq, sk])?)?;
    }
    scores.softmax(2)?.bmm(v)
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
    cross_entropy(logits, &one_hot(target_ids, logits.shape()[1])?)
}
