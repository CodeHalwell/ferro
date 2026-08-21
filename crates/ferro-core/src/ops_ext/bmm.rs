//! `bmm` batched matrix multiply. self (b,m,k) @ other (b,k,n) -> (b,m,n).
//! Backward per batch: dA = dC @ B^T, dB = A^T @ dC.
//! Every GEMM (forward and both backward products) routes through the CPU
//! backend's `matmul_batch` as a single call, so bmm rides whatever kernel is
//! installed there (e.g. ferro-fastcpu's one-thread::scope batch-parallel
//! path) instead of spawning a thread pool per batch element. The backend
//! takes plain row-major operands, so the backward pass materializes every
//! batch's B^T/A^T transpose into one contiguous scratch buffer before the
//! single batched matmul call.

use crate::device::Device;
use crate::dispatch;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor> {
        if self.device() != other.device() {
            return Err(Error::DeviceMismatch { op: "bmm", lhs: self.device(), rhs: other.device() });
        }
        if self.ndim() != 3 || other.ndim() != 3 {
            return Err(Error::Unsupported { op: "bmm", msg: "inputs must be rank 3".into() });
        }
        let (a_shape, b_shape) = (self.shape(), other.shape());
        let (batch, m, k) = (a_shape[0], a_shape[1], a_shape[2]);
        let n = b_shape[2];
        if b_shape[0] != batch || b_shape[1] != k {
            return Err(Error::ShapeMismatch {
                op: "bmm",
                lhs: a_shape.to_vec(),
                rhs: b_shape.to_vec(),
            });
        }

        let cpu = dispatch::backend_for(Device::Cpu)?;
        let a_data = self.to_vec();
        let b_data = other.to_vec();
        let c_data = cpu.matmul_batch(&a_data, &b_data, batch, m, k, n);
        let out = Tensor::from_vec(c_data, &[batch, m, n])?;

        let a = self.clone();
        let b = other.clone();
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let cpu = dispatch::backend_for(Device::Cpu).expect("cpu backend is always registered");
            let g_data = g.to_vec();
            let a_data = a.to_vec();
            let b_data = b.to_vec();

            // dA[b] = dC[b] @ B[b]^T -> (m,k). Materialize every batch's
            // B^T into one contiguous [batch,n,k] buffer, then a single
            // matmul_batch call (O(batch*kn) transpose against the
            // O(batch*mkn) GEMMs).
            let mut bt = vec![0.0f32; batch * n * k];
            for bi in 0..batch {
                let (bo, bto) = (bi * k * n, bi * n * k);
                for p in 0..k {
                    for j in 0..n {
                        bt[bto + j * k + p] = b_data[bo + p * n + j];
                    }
                }
            }
            let da = cpu.matmul_batch(&g_data, &bt, batch, m, n, k);

            // dB[b] = A[b]^T @ dC[b] -> (k,n). Same idea: one [batch,k,m]
            // buffer of A^T.
            let mut at = vec![0.0f32; batch * k * m];
            for bi in 0..batch {
                let (ao, ato) = (bi * m * k, bi * k * m);
                for i in 0..m {
                    for p in 0..k {
                        at[ato + p * m + i] = a_data[ao + i * k + p];
                    }
                }
            }
            let db = cpu.matmul_batch(&at, &g_data, batch, k, m, n);

            vec![
                Tensor::from_vec(da, &[batch, m, k]).unwrap(),
                Tensor::from_vec(db, &[batch, k, n]).unwrap(),
            ]
        }))
    }
}
