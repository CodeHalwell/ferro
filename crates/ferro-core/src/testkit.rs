//! Test utilities usable from integration tests and downstream op crates.
//! Kept in the library (not behind `#[cfg(test)]`) so every `tests/op_*.rs`
//! file can call the same finite-difference checker with one line.

use crate::tensor::Tensor;

/// Central-difference gradient check. `f` builds a scalar loss from the given
/// leaves; each leaf is perturbed elementwise and the numerical dL/dx compared
/// against the autograd gradient. Panics on mismatch. Uses an absolute+relative
/// band because f32 central differences on larger-magnitude losses are noisy.
pub fn grad_check<F>(inputs: &[Tensor], f: F)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let leaves: Vec<Tensor> = inputs.iter().map(|t| t.requires_grad_(true)).collect();
    f(&leaves).backward();

    let eps = 4e-3f32;
    for (li, leaf) in leaves.iter().enumerate() {
        let base = leaf.to_vec();
        let analytic = leaf.grad().expect("leaf should have a grad").to_vec();
        for i in 0..base.len() {
            let mut up = base.clone();
            up[i] += eps;
            let mut dn = base.clone();
            dn[i] -= eps;
            let mut plus = leaves.clone();
            plus[li] = Tensor::from_vec(up, leaf.shape()).unwrap();
            let mut minus = leaves.clone();
            minus[li] = Tensor::from_vec(dn, leaf.shape()).unwrap();
            let numeric = (f(&plus).item() - f(&minus).item()) / (2.0 * eps);
            let tol = 1e-2 + 2e-2 * analytic[i].abs();
            assert!(
                (analytic[i] - numeric).abs() <= tol,
                "grad mismatch input {li} elem {i}: analytic {} vs numeric {} (tol {tol})",
                analytic[i],
                numeric
            );
        }
    }
}
