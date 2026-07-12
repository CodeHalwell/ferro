//! `bmm` batched matrix multiply. self (b,m,k) @ other (b,k,n) -> (b,m,n).
//! Backward per batch: dA = dC @ B^T, dB = A^T @ dC.
//! Every per-batch GEMM (forward and backward) routes through the CPU
//! backend's `matmul`, so bmm rides whatever kernel is installed there (e.g.
//! ferro-fastcpu) instead of a fixed naive loop. The backend takes plain
//! row-major operands, so the backward pass materializes the small B^T/A^T
//! transposes into scratch buffers reused across the batch loop.

use crate::device::Device;
use crate::dispatch::{self, Backend};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

// C[b] = A[b] @ B[b] over flat row-major buffers, one backend matmul per batch.
fn batched_matmul(
    cpu: &dyn Backend,
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
        let c_bi = cpu.matmul(&a[ao..ao + m * k], &b[bo..bo + k * n], m, k, n);
        c[co..co + m * n].copy_from_slice(&c_bi);
    }
    c
}

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
        let c_data = batched_matmul(cpu.as_ref(), &a_data, &b_data, batch, m, k, n);
        let out = Tensor::from_vec(c_data, &[batch, m, n])?;

        let a = self.clone();
        let b = other.clone();
        Ok(out.record_fn(vec![self.clone(), other.clone()], move |g| {
            let cpu = dispatch::backend_for(Device::Cpu).expect("cpu backend is always registered");
            let g_data = g.to_vec();
            let a_data = a.to_vec();
            let b_data = b.to_vec();

            // dA[b] = dC[b] @ B[b]^T -> (m,k). B[b]^T is (n,k); transpose into
            // a scratch buffer reused across batches (O(kn) vs. the O(mkn) GEMM).
            let mut bt = vec![0.0f32; n * k];
            let mut da = vec![0.0f32; batch * m * k];
            for bi in 0..batch {
                let (go, bo, ao) = (bi * m * n, bi * k * n, bi * m * k);
                for p in 0..k {
                    for j in 0..n {
                        bt[j * k + p] = b_data[bo + p * n + j];
                    }
                }
                let da_bi = cpu.matmul(&g_data[go..go + m * n], &bt, m, n, k);
                da[ao..ao + m * k].copy_from_slice(&da_bi);
            }

            // dB[b] = A[b]^T @ dC[b] -> (k,n). A[b]^T is (k,m); same reuse idea.
            let mut at = vec![0.0f32; k * m];
            let mut db = vec![0.0f32; batch * k * n];
            for bi in 0..batch {
                let (go, ao, bo) = (bi * m * n, bi * m * k, bi * k * n);
                for i in 0..m {
                    for p in 0..k {
                        at[p * m + i] = a_data[ao + i * k + p];
                    }
                }
                let db_bi = cpu.matmul(&at, &g_data[go..go + m * n], k, m, n);
                db[bo..bo + k * n].copy_from_slice(&db_bi);
            }

            vec![
                Tensor::from_vec(da, &[batch, m, k]).unwrap(),
                Tensor::from_vec(db, &[batch, k, n]).unwrap(),
            ]
        }))
    }
}
