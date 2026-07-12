//! Inverted dropout, counter-based (docs/CAPABILITY.md S7). Mask element i
//! comes from `Philox::uniform_at(offset, i)` keyed by `seed`, so a mask
//! recomputed under checkpoint recompute is bitwise identical regardless of
//! thread count or evaluation order - `Rng` (sequential xorshift128+) cannot
//! give that guarantee. (seed, offset) are explicit caller-supplied
//! parameters for now; per-graph RNG state and train/eval mode plumbing on
//! `Module` is future work (CAPABILITY.md S7/S8).
//!
//! Host fallback: like other composite ops, forward runs on the host via
//! `to_vec` and returns a cpu tensor even for device inputs.

use crate::error::{Error, Result};
use crate::philox::Philox;
use crate::tensor::{raw_binary, Tensor};

impl Tensor {
    pub fn dropout(&self, p: f32, train: bool, seed: u64, offset: u64) -> Result<Tensor> {
        if !(0.0..1.0).contains(&p) {
            return Err(Error::Unsupported { op: "dropout", msg: format!("p must be in [0, 1), got {p}") });
        }
        if !train || p == 0.0 {
            return Ok(self.clone());
        }

        let philox = Philox::new(seed);
        let scale = 1.0 / (1.0 - p);
        let x = self.to_vec();
        let mask_data: Vec<f32> = (0..x.len() as u64)
            .map(|i| if philox.uniform_at(offset, i) < p { 0.0 } else { scale })
            .collect();
        let out_data: Vec<f32> = x.iter().zip(&mask_data).map(|(&xi, &mi)| xi * mi).collect();
        let out = Tensor::from_vec(out_data, self.shape())?;

        let mask = Tensor::from_vec(mask_data, self.shape())?;
        Ok(out.record_fn(vec![self.clone()], move |g| {
            vec![raw_binary("dropout_bw", g, &mask, |gg, m| gg * m).unwrap()]
        }))
    }
}
