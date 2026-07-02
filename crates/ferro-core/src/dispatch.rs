//! Kernel dispatch table: named, structured kernels route through swappable
//! function pointers so a backend crate can override them without touching
//! core (the ATen dispatch idea in miniature). Elementwise ops still run as
//! inline CPU closures; they migrate here when a second device arrives.

use std::sync::RwLock;

/// Row-major (m,k) @ (k,n) -> (m,n).
pub type MatmulKernel = fn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;

static MATMUL: RwLock<MatmulKernel> = RwLock::new(naive_matmul);

/// Install a replacement matmul kernel (e.g. a blocked/threaded backend).
/// Applies process-wide to all CPU tensors from the next call onward.
pub fn set_matmul_kernel(kernel: MatmulKernel) {
    *MATMUL.write().unwrap() = kernel;
}

pub(crate) fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    (MATMUL.read().unwrap())(a, b, m, k, n)
}

/// Reference implementation: ikj loop order with a zero-skip that helps the
/// ReLU-sparse gradients common in backward passes.
pub fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            let brow = p * n;
            let orow = i * n;
            for j in 0..n {
                out[orow + j] += aip * b[brow + j];
            }
        }
    }
    out
}
