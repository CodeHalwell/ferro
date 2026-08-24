//! Test utilities usable from integration tests and downstream op crates.
//! Kept in the library (not behind `#[cfg(test)]`) so every `tests/op_*.rs`
//! file can call the same finite-difference checker with one line.

use crate::tensor::Tensor;

/// Central-difference gradient check. `f` builds a scalar loss from the given
/// leaves; each leaf is perturbed elementwise and the numerical dL/dx compared
/// against the autograd gradient. Panics on mismatch. Uses an absolute+relative
/// band because f32 central differences on larger-magnitude losses are noisy.
/// Tolerance profile for `grad_check_opts`. `Default` matches the historical
/// loose band; `Strict` is tighter and meant for smooth composite ops whose
/// closed-form backwards are exact.
#[derive(Clone, Copy)]
pub enum GradTol {
    Default,
    Strict,
}

impl GradTol {
    fn eps(&self) -> f32 {
        match self {
            GradTol::Default => 4e-3,
            GradTol::Strict => 1e-3,
        }
    }
    fn tol(&self, analytic: f32) -> f32 {
        match self {
            GradTol::Default => 1e-2 + 2e-2 * analytic.abs(),
            GradTol::Strict => 5e-4 + 1e-3 * analytic.abs(),
        }
    }
}

/// Central-difference gradient check with a tolerance profile.
fn grad_check_with<F>(inputs: &[Tensor], f: F, tol: GradTol)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let leaves: Vec<Tensor> = inputs
        .iter()
        .map(|t| {
            t.requires_grad_(true)
                .expect("grad_check inputs are leaves")
        })
        .collect();
    f(&leaves).backward();

    let eps = tol.eps();
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
            let lim = tol.tol(analytic[i]);
            assert!(
                (analytic[i] - numeric).abs() <= lim,
                "grad mismatch input {li} elem {i}: analytic {} vs numeric {} (tol {lim})",
                analytic[i],
                numeric
            );
        }
    }
}

pub fn grad_check<F>(inputs: &[Tensor], f: F)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    grad_check_with(inputs, f, GradTol::Default);
}

pub fn grad_check_strict<F>(inputs: &[Tensor], f: F)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    grad_check_with(inputs, f, GradTol::Strict);
}
