//! `log_softmax` over one dimension. Numerically stable via the log-sum-exp
//! trick: along each slice `m = max(x)`, `lse = m + log(sum(exp(x - m)))`, and
//! `y_i = x_i - lse`. Backward: with `sm_i = exp(y_i)` (softmax), the gradient is
//! `dx_i = g_i - sm_i * sum_k g_k`, computed per slice over `dim`.

use crate::error::{Error, Result};
use crate::reduce::pairwise_sum_strided;
use crate::tensor::Tensor;

impl Tensor {
    pub fn log_softmax(&self, dim: usize) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape { op: "log_softmax", msg: format!("dim {dim} out of range for rank {ndim}") });
        }
        let shape = self.shape().to_vec();
        let x = self.to_vec();
        let n = shape[dim];
        let stride = shape[dim + 1..].iter().product::<usize>();
        let outer = shape[..dim].iter().product::<usize>();

        let mut y = vec![0.0f32; x.len()];
        let mut exps = vec![0.0f32; x.len()];
        for o in 0..outer {
            for i in 0..stride {
                let base = o * n * stride + i;
                let mut m = f32::NEG_INFINITY;
                for k in 0..n {
                    m = m.max(x[base + k * stride]);
                }
                for k in 0..n {
                    exps[base + k * stride] = (x[base + k * stride] - m).exp();
                }
                let sum = pairwise_sum_strided(&exps, base, n, stride);
                let lse = m + sum.ln();
                for k in 0..n {
                    y[base + k * stride] = x[base + k * stride] - lse;
                }
            }
        }

        if !self.requires_grad() {
            return Tensor::from_vec(y, &shape);
        }
        // Save softmax = exp(log_softmax output) for the backward; from_vec
        // already yields a fresh detached leaf.
        let sm_data: Vec<f32> = y.iter().map(|&v| v.exp()).collect();
        let out = Tensor::from_vec(y, &shape).unwrap();
        let sm = Tensor::from_vec(sm_data, &shape).unwrap();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gv = g.to_vec();
            let smv = sm.to_vec();
            let mut dx = vec![0.0f32; gv.len()];
            for o in 0..outer {
                for i in 0..stride {
                    let base = o * n * stride + i;
                    let sum_g = pairwise_sum_strided(&gv, base, n, stride);
                    for k in 0..n {
                        let idx = base + k * stride;
                        dx[idx] = gv[idx] - smv[idx] * sum_g;
                    }
                }
            }
            vec![Tensor::from_vec(dx, &shape).unwrap()]
        }))
    }
}
