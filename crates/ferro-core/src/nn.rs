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

    /// Parameters with torch-style names ("weight", "bias", "0.weight" inside
    /// a Sequential); the contract behind state_dict save/load.
    fn named_parameters(&self) -> Vec<(String, Param)>;

    /// Switch between training and evaluation behaviour (dropout masking,
    /// BatchNorm running stats). Default: stateless layer, nothing to do.
    fn set_training(&self, _training: bool) {}

    fn parameters(&self) -> Vec<Param> {
        self.named_parameters()
            .into_iter()
            .map(|(_, p)| p)
            .collect()
    }
}

/// Put a module tree into training mode.
pub fn train(m: &dyn Module) {
    m.set_training(true);
}

/// Put a module tree into evaluation mode.
pub fn eval(m: &dyn Module) {
    m.set_training(false);
}

/// Weight-initialization schemes. `std` gives the standard deviation of the
/// normal distribution to draw each weight from.
///
/// - Normal(std): plain N(0, std^2), the transformer-style small-init default.
/// - Kaiming: He et al. 2015 normal init for relu nets, std = sqrt(2/fan_in).
/// - Xavier: Glorot normal, std = sqrt(2/(fan_in+fan_out)).
#[derive(Clone, Copy, Debug)]
pub enum Init {
    Normal(f32),
    Kaiming,
    Xavier,
}

impl Init {
    pub fn std(&self, fan_in: usize, fan_out: usize) -> f32 {
        match *self {
            Init::Normal(s) => s,
            Init::Kaiming => (2.0 / fan_in as f32).sqrt(),
            Init::Xavier => (2.0 / (fan_in + fan_out) as f32).sqrt(),
        }
    }

    pub fn fill(&self, rng: &Rng, shape: &[usize], fan_in: usize, fan_out: usize) -> Tensor {
        let s = self.std(fan_in, fan_out);
        let data: Vec<f32> = (0..crate::shape::numel(shape))
            .map(|_| rng.normal() * s)
            .collect();
        Tensor::from_vec(data, shape).expect("init shape is valid")
    }
}

/// Save a module's parameters as a safetensors state dict.
pub fn save_module<P: AsRef<std::path::Path>>(path: P, module: &dyn Module) -> Result<()> {
    let named = module.named_parameters();
    let tensors: Vec<(String, Tensor)> =
        named.iter().map(|(n, p)| (n.clone(), p.tensor())).collect();
    let refs: Vec<(&str, &Tensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    crate::safetensors::save_safetensors(path, &refs)
}

/// Load a safetensors state dict into a module, strictly (torch semantics):
/// every parameter must be present with a matching shape, and every tensor in
/// the file must correspond to a parameter.
pub fn load_module<P: AsRef<std::path::Path>>(path: P, module: &dyn Module) -> Result<()> {
    let mut loaded = crate::safetensors::load_safetensors(path)?;
    for (name, param) in module.named_parameters() {
        let pos = loaded
            .iter()
            .position(|(n, _)| *n == name)
            .ok_or_else(|| Error::Format {
                op: "load_module",
                msg: format!("state dict is missing parameter {name:?}"),
            })?;
        let (_, t) = loaded.swap_remove(pos);
        let want = param.tensor();
        if t.shape() != want.shape() || t.dtype() != want.dtype() {
            return Err(Error::Format {
                op: "load_module",
                msg: format!(
                    "parameter {name:?}: expected {} {:?}, file has {} {:?}",
                    want.dtype(),
                    want.shape(),
                    t.dtype(),
                    t.shape()
                ),
            });
        }
        param.set(t);
    }
    if let Some((name, _)) = loaded.first() {
        return Err(Error::Format {
            op: "load_module",
            msg: format!("state dict has unexpected tensor {name:?}"),
        });
    }
    Ok(())
}

/// Affine layer `y = x @ W + b` with He-initialized weights.
pub struct Linear {
    weight: Param,
    bias: Param,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, rng: &Rng) -> Linear {
        let scale = (2.0 / in_features as f32).sqrt();
        let w: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.normal() * scale)
            .collect();
        let weight = Param::new(Tensor::from_vec(w, &[in_features, out_features]).unwrap());
        let bias = Param::new(Tensor::zeros(&[out_features]));
        Linear { weight, bias }
    }

    /// Same layer with a caller-chosen weight-init scheme.
    pub fn with_init(in_features: usize, out_features: usize, rng: &Rng, init: Init) -> Linear {
        let weight =
            Param::new(init.fill(rng, &[in_features, out_features], in_features, out_features));
        let bias = Param::new(Tensor::zeros(&[out_features]));
        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight.tensor())?.add(&self.bias.tensor())
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![
            ("weight".into(), self.weight.clone()),
            ("bias".into(), self.bias.clone()),
        ]
    }
}

pub struct Relu;

impl Module for Relu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.relu())
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        Vec::new()
    }
}

pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.sigmoid())
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        Vec::new()
    }
}

pub struct Gelu;

impl Module for Gelu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.gelu())
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
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
        LayerNorm {
            gamma,
            beta,
            eps: 1e-5,
        }
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
        // The eps scalar must live on x's device for device-resident inputs.
        let eps = Tensor::full_on(&[], self.eps, x.device())?;
        let norm = centered.div(&var.add(&eps)?.sqrt())?;
        norm.mul(&self.gamma.tensor())?.add(&self.beta.tensor())
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![
            ("weight".into(), self.gamma.clone()),
            ("bias".into(), self.beta.clone()),
        ]
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
        RmsNorm {
            gamma: Param::new(Tensor::ones(&[dim])),
            eps: 1e-5,
        }
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last = x.ndim() - 1;
        let ms = x.mul(x)?.mean_dim(last, true)?;
        // The eps scalar must live on x's device for device-resident inputs.
        let eps = Tensor::full_on(&[], self.eps, x.device())?;
        x.div(&ms.add(&eps)?.sqrt())?.mul(&self.gamma.tensor())
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![("weight".into(), self.gamma.clone())]
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
        Embedding {
            weight: Param::new(Tensor::from_vec(w, &[num_embeddings, dim]).unwrap()),
            dim,
        }
    }
}

impl Module for Embedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let flat = if ids.ndim() == 1 {
            ids.clone()
        } else {
            ids.reshape(&[ids.numel()])?
        };
        let out = crate::ops_ext::embedding(&self.weight.tensor(), &flat)?;
        let mut shape = ids.shape().to_vec();
        shape.push(self.dim);
        out.reshape(&shape)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![("weight".into(), self.weight.clone())]
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
        return Err(Error::DtypeMismatch {
            op: "one_hot",
            expected: DType::I64,
            got: ids.dtype(),
        });
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
    // Land on the ids' device so cross_entropy_indices works with device-
    // resident targets; the one-hot build itself is host-side (ids were
    // downloaded once above), then transferred like any other leaf.
    Tensor::from_vec(data, &[idx.len(), classes])?.to_device(ids.device())
}

/// Scaled dot-product attention over `[batch, seq, head_dim]` inputs (fold
/// heads into batch before calling): softmax(q k^T / sqrt(d) + mask) v, with
/// an optional causal mask (position i attends to j <= i; masked scores get
/// -1e9 rather than -inf so softmax stays NaN-free). Composed from autograd
/// ops, so gradients flow to q, k, and v without a custom backward.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
) -> Result<Tensor> {
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.ndim() != 3 {
            return Err(Error::InvalidShape {
                op: "scaled_dot_product_attention",
                msg: format!(
                    "{name} must be 3-D [batch, seq, head_dim], got {:?}",
                    t.shape()
                ),
            });
        }
    }
    let (b, sq, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let sk = k.shape()[1];
    if k.shape() != [b, sk, d] || v.shape()[0] != b || v.shape()[1] != sk {
        return Err(Error::InvalidShape {
            op: "scaled_dot_product_attention",
            msg: format!(
                "incompatible shapes q {:?}, k {:?}, v {:?}",
                q.shape(),
                k.shape(),
                v.shape()
            ),
        });
    }
    // The scale scalar must live on q's device for device-resident attention.
    let scale = Tensor::full_on(&[], 1.0 / (d as f32).sqrt(), q.device())?;
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
        // The mask must live on the scores' device for device-resident q/k.
        let mask = Tensor::from_vec(m, &[sq, sk])?.to_device(scores.device())?;
        scores = scores.add(&mask)?;
    }
    scores.softmax(2)?.bmm(v)
}

/// Multi-head self-attention over `[batch, seq, dim]` (LLaMA-shaped: four
/// bias-free square projections named q_proj/k_proj/v_proj/o_proj, optional
/// half-split RoPE on q and k, causal masking). Heads fold into the batch dim
/// around the shared `scaled_dot_product_attention`.
pub struct MultiHeadAttention {
    q_proj: Param,
    k_proj: Param,
    v_proj: Param,
    o_proj: Param,
    heads: usize,
    causal: bool,
    rope_base: Option<f32>,
}

impl MultiHeadAttention {
    pub fn new(dim: usize, heads: usize, causal: bool, rng: &Rng) -> Result<MultiHeadAttention> {
        if heads == 0 || dim % heads != 0 {
            return Err(Error::InvalidShape {
                op: "multi_head_attention",
                msg: format!("dim {dim} is not divisible into {heads} heads"),
            });
        }
        let scale = 1.0 / (dim as f32).sqrt();
        let proj = || {
            let w: Vec<f32> = (0..dim * dim).map(|_| rng.normal() * scale).collect();
            Param::new(Tensor::from_vec(w, &[dim, dim]).unwrap())
        };
        Ok(MultiHeadAttention {
            q_proj: proj(),
            k_proj: proj(),
            v_proj: proj(),
            o_proj: proj(),
            heads,
            causal,
            rope_base: None,
        })
    }

    /// Apply RoPE to q and k before attention (positions 0..seq).
    pub fn with_rope(mut self, base: f32) -> MultiHeadAttention {
        self.rope_base = Some(base);
        self
    }

    /// `[b, s, d] -> [b*h, s, d/h]`: project, split heads, fold into batch.
    fn heads_in(&self, x: &Tensor, w: &Param, b: usize, s: usize, d: usize) -> Result<Tensor> {
        let hd = d / self.heads;
        let p = x.reshape(&[b * s, d])?.matmul(&w.tensor())?;
        p.reshape(&[b, s, self.heads, hd])?
            .transpose(1, 2)?
            .reshape(&[b * self.heads, s, hd])
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(Error::InvalidShape {
                op: "multi_head_attention",
                msg: format!("input must be 3-D [batch, seq, dim], got {:?}", x.shape()),
            });
        }
        let (b, s, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if d != self.q_proj.tensor().shape()[0] {
            return Err(Error::InvalidShape {
                op: "multi_head_attention",
                msg: format!(
                    "input dim {d} does not match projection dim {}",
                    self.q_proj.tensor().shape()[0]
                ),
            });
        }
        let mut q = self.heads_in(x, &self.q_proj, b, s, d)?;
        let mut k = self.heads_in(x, &self.k_proj, b, s, d)?;
        let v = self.heads_in(x, &self.v_proj, b, s, d)?;
        if let Some(base) = self.rope_base {
            let pos = Tensor::arange(s as i64);
            q = q.rope(&pos, base)?;
            k = k.rope(&pos, base)?;
        }
        let attn = scaled_dot_product_attention(&q, &k, &v, self.causal)?;
        let merged = attn
            .reshape(&[b, self.heads, s, d / self.heads])?
            .transpose(1, 2)?
            .reshape(&[b * s, d])?;
        merged.matmul(&self.o_proj.tensor())?.reshape(&[b, s, d])
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        vec![
            ("q_proj".into(), self.q_proj.clone()),
            ("k_proj".into(), self.k_proj.clone()),
            ("v_proj".into(), self.v_proj.clone()),
            ("o_proj".into(), self.o_proj.clone()),
        ]
    }
}

/// Pre-norm transformer block: `x + attn(norm1(x))`, then `x + mlp(norm2(x))`
/// with a Gelu MLP at 4x width. The building block for milestone M3.
pub struct TransformerBlock {
    norm1: RmsNorm,
    attn: MultiHeadAttention,
    norm2: RmsNorm,
    up: Linear,
    down: Linear,
}

impl TransformerBlock {
    pub fn new(dim: usize, heads: usize, rng: &Rng) -> Result<TransformerBlock> {
        Ok(TransformerBlock {
            norm1: RmsNorm::new(dim),
            attn: MultiHeadAttention::new(dim, heads, true, rng)?.with_rope(10000.0),
            norm2: RmsNorm::new(dim),
            up: Linear::new(dim, 4 * dim, rng),
            down: Linear::new(4 * dim, dim, rng),
        })
    }

    /// The MLP runs per token: flatten `[b, s, d]` to `[b*s, d]` for the 2-D
    /// Linear layers, then restore.
    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().to_vec();
        let d = shape[2];
        let flat = x.reshape(&[shape[0] * shape[1], d])?;
        let out = self.down.forward(&self.up.forward(&flat)?.gelu())?;
        out.reshape(&shape)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(Error::InvalidShape {
                op: "transformer_block",
                msg: format!("input must be 3-D [batch, seq, dim], got {:?}", x.shape()),
            });
        }
        let h = x.add(&self.attn.forward(&self.norm1.forward(x)?)?)?;
        h.add(&self.mlp(&self.norm2.forward(&h)?)?)
    }

    fn named_parameters(&self) -> Vec<(String, Param)> {
        let mut out = Vec::new();
        for (prefix, m) in [
            ("norm1", &self.norm1 as &dyn Module),
            ("attn", &self.attn),
            ("norm2", &self.norm2),
            ("mlp.up", &self.up),
            ("mlp.down", &self.down),
        ] {
            out.extend(
                m.named_parameters()
                    .into_iter()
                    .map(|(n, p)| (format!("{prefix}.{n}"), p)),
            );
        }
        out
    }
}

/// Cross-entropy for `[batch, classes]` logits against I64 class ids `[batch]`
/// (like PyTorch's `F.cross_entropy` with integer targets): one-hot encode the
/// ids, then reuse `cross_entropy`.
pub fn cross_entropy_indices(logits: &Tensor, target_ids: &Tensor) -> Result<Tensor> {
    if logits.ndim() != 2 {
        return Err(Error::InvalidShape {
            op: "cross_entropy_indices",
            msg: format!(
                "logits must be 2-D [batch, classes], got {:?}",
                logits.shape()
            ),
        });
    }
    if target_ids.numel() != logits.shape()[0] {
        return Err(Error::InvalidShape {
            op: "cross_entropy_indices",
            msg: format!(
                "{} target ids for batch of {}",
                target_ids.numel(),
                logits.shape()[0]
            ),
        });
    }
    cross_entropy(logits, &one_hot(target_ids, logits.shape()[1])?)
}
