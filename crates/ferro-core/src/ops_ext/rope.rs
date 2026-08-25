//! Rotary position embedding (RoPE), half-split convention (LLaMA/HF): the
//! last dim is split into halves (x1, x2) and each pair (x1[j], x2[j]) is
//! rotated by pos * base^(-2j/d):
//!   y1 = x1*cos - x2*sin,  y2 = x2*cos + x1*sin.
//! Input is [..., seq, head_dim] with head_dim even; `positions` is a 1-D I64
//! tensor of length seq (explicit so KV-cache decode can offset positions).
//! The rotation is orthogonal and linear, so backward applies the inverse
//! rotation (negated sin) to the incoming grad.
//!
//! cos/sin tables are cached per (seq_len, head_dim, base) config: the
//! training/prefill path (`rope_cached`, positions 0..seq) hits the cache
//! after the first call with no table rebuild and no position traffic at
//! all. The rotation itself is host-composed because dispatch has no rope or
//! permute kernel yet, so tables live as host slices (uploading them would
//! only add a per-call download); when a device kernel lands they can move
//! on-device without changing this cache.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

struct RopeTables {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

// Keyed by (seq_len, head_dim, base bits); holds only the canonical
// positions 0..seq tables. Entries are immutable and never evicted - each
// costs 2 * seq * (head_dim/2) f32s, so process-lifetime retention is bounded
// by the number of distinct configs a run uses (one per model shape).
static TABLES: OnceLock<Mutex<HashMap<(usize, usize, u32), Arc<RopeTables>>>> = OnceLock::new();

fn tables_for(seq: usize, dim: usize, base: f32, pos: &[i64]) -> Arc<RopeTables> {
    let canonical = pos.len() == seq && pos.iter().enumerate().all(|(i, &p)| p == i as i64);
    let key = (seq, dim, base.to_bits());
    let mut cache = TABLES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if canonical {
        if let Some(t) = cache.get(&key) {
            return t.clone();
        }
    }
    let half = dim / 2;
    let mut cos = vec![0.0f32; seq * half];
    let mut sin = vec![0.0f32; seq * half];
    for s in 0..seq {
        for j in 0..half {
            let theta = pos[s] as f32 * base.powf(-2.0 * j as f32 / dim as f32);
            cos[s * half + j] = theta.cos();
            sin[s * half + j] = theta.sin();
        }
    }
    let t = Arc::new(RopeTables { cos, sin });
    if canonical {
        cache.insert(key, t.clone());
    }
    t
}

fn validate(input: &Tensor, op: &'static str) -> Result<(Vec<usize>, usize, usize)> {
    let ndim = input.ndim();
    if ndim < 2 {
        return Err(Error::InvalidShape {
            op,
            msg: format!(
                "input must be at least 2-D [seq, head_dim], got {:?}",
                input.shape()
            ),
        });
    }
    let shape = input.shape().to_vec();
    let (seq, dim) = (shape[ndim - 2], shape[ndim - 1]);
    if dim % 2 != 0 {
        return Err(Error::InvalidShape {
            op,
            msg: format!("head_dim {dim} must be even"),
        });
    }
    Ok((shape, seq, dim))
}

impl Tensor {
    pub fn rope(&self, positions: &Tensor, base: f32) -> Result<Tensor> {
        let (shape, seq, dim) = validate(self, "rope")?;
        if positions.dtype() != DType::I64 {
            return Err(Error::DtypeMismatch {
                op: "rope",
                expected: DType::I64,
                got: positions.dtype(),
            });
        }
        if positions.ndim() != 1 || positions.shape()[0] != seq {
            return Err(Error::InvalidShape {
                op: "rope",
                msg: format!(
                    "positions must be 1-D of length {seq}, got shape {:?}",
                    positions.shape()
                ),
            });
        }

        // Position values must be read once for correctness (arbitrary
        // KV-cache offsets are allowed); the config cache absorbs the rebuild
        // whenever they turn out to be the canonical 0..seq.
        let pos = positions.to_vec_i64();
        let tabs = tables_for(seq, dim, base, &pos);
        self.apply(tabs, &shape)
    }

    /// RoPE over the implicit positions 0..seq: no position tensor exists at
    /// the call site, so after the first call for a (seq_len, head_dim, base)
    /// config there is zero table computation and zero position traffic -
    /// positions effectively live inside the device-resident pipeline rather
    /// than round-tripping from host storage each step.
    pub fn rope_cached(&self, base: f32) -> Result<Tensor> {
        let (shape, seq, dim) = validate(self, "rope_cached")?;
        let pos: Vec<i64> = (0..seq as i64).collect();
        let tabs = tables_for(seq, dim, base, &pos);
        self.apply(tabs, &shape)
    }

    fn apply(&self, tabs: Arc<RopeTables>, shape: &[usize]) -> Result<Tensor> {
        let shape = shape.to_vec();
        let ndim = shape.len();
        let (seq, dim) = (shape[ndim - 2], shape[ndim - 1]);
        let half = dim / 2;
        let batch: usize = shape[..ndim - 2].iter().product();
        let x = self.to_vec();
        let y = rotate(&x, &tabs.cos, &tabs.sin, batch, seq, half, 1.0);
        // RoPE is host-composed; return to the input's device so chained
        // device-resident ops (attention, matmul) stay on-device.
        let out = Tensor::from_vec(y, &shape)?.to_device(self.device())?;
        if !self.requires_grad() {
            return Ok(out);
        }
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let dx = rotate(&g.to_vec(), &tabs.cos, &tabs.sin, batch, seq, half, -1.0);
            vec![Tensor::from_vec(dx, &shape).unwrap()]
        }))
    }
}

fn rotate(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    seq: usize,
    half: usize,
    sign: f32,
) -> Vec<f32> {
    let dim = 2 * half;
    let mut y = vec![0.0f32; x.len()];
    for b in 0..batch {
        for s in 0..seq {
            let row = (b * seq + s) * dim;
            for j in 0..half {
                let (c, sn) = (cos[s * half + j], sign * sin[s * half + j]);
                let (x1, x2) = (x[row + j], x[row + half + j]);
                y[row + j] = x1 * c - x2 * sn;
                y[row + half + j] = x2 * c + x1 * sn;
            }
        }
    }
    y
}
