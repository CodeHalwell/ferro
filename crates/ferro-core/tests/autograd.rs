use ferro_core::{Rng, Tensor};

fn assert_close(a: f32, b: f32, tol: f32, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
}

/// Relative closeness: f32 central differences on large-magnitude losses carry
/// real rounding noise, so compare with an absolute + relative band.
fn assert_close_rel(analytic: f32, numeric: f32, what: &str) {
    let tol = 1e-2 + 2e-2 * analytic.abs();
    assert!((analytic - numeric).abs() <= tol, "{what}: {analytic} vs {numeric} (tol {tol})");
}

/// Central-difference gradient check: perturb each input element and compare the
/// numerical dL/dx against the autograd grad. `f` builds the scalar loss from a
/// list of leaf tensors; `inputs` are the leaves to check.
fn grad_check<F>(inputs: &[Tensor], f: F)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let leaves: Vec<Tensor> = inputs.iter().map(|t| t.requires_grad_(true)).collect();
    let loss = f(&leaves);
    loss.backward();

    let eps = 4e-3f32;
    for (li, leaf) in leaves.iter().enumerate() {
        let base = leaf.to_vec();
        let analytic = leaf.grad().expect("leaf should have grad").to_vec();
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
            assert_close_rel(analytic[i], numeric, &format!("input {li} elem {i}"));
        }
    }
}

#[test]
fn add_mul_broadcast() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    grad_check(&[a, b], |t| t[0].add(&t[1]).unwrap().mul(&t[0]).unwrap().sum());
}

#[test]
fn sub_div() {
    let a = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| t[0].sub(&t[1]).unwrap().div(&t[1]).unwrap().sum());
}

#[test]
fn matmul_relu_chain() {
    let x = Tensor::from_vec(vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.5], &[2, 3]).unwrap();
    let w = Tensor::from_vec(vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6], &[3, 2]).unwrap();
    grad_check(&[x, w], |t| t[0].matmul(&t[1]).unwrap().relu().sum());
}

#[test]
fn exp_sigmoid_mean() {
    let a = Tensor::from_vec(vec![-0.5, 0.2, 1.0, -1.3], &[4]).unwrap();
    grad_check(&[a], |t| t[0].sigmoid().exp().mean());
}

#[test]
fn transpose_matmul() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    grad_check(&[a], |t| t[0].transpose(0, 1).unwrap().matmul(&t[0]).unwrap().sum());
}

#[test]
fn reused_leaf_accumulates() {
    // x used twice: grad of x*x at x is 2x.
    let x = Tensor::from_vec(vec![3.0, -2.0], &[2]).unwrap().requires_grad_(true);
    let y = x.mul(&x).unwrap().sum();
    y.backward();
    let g = x.grad().unwrap().to_vec();
    assert_close(g[0], 6.0, 1e-4, "d(x^2) at 3");
    assert_close(g[1], -4.0, 1e-4, "d(x^2) at -2");
}

#[test]
#[should_panic(expected = "scalar output")]
fn backward_on_non_scalar_panics() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap().requires_grad_(true);
    a.mul(&a).unwrap().backward();
}

#[test]
fn linear_regression_converges() {
    // Fit y = x @ w_true + b_true with plain SGD; loss must drop sharply.
    let rng = Rng::new(42);
    let n = 64;
    let xs: Vec<f32> = (0..n * 2).map(|_| rng.normal()).collect();
    let x = Tensor::from_vec(xs, &[n, 2]).unwrap();
    let w_true = Tensor::from_vec(vec![2.0, -3.0], &[2, 1]).unwrap();
    let y = x.matmul(&w_true).unwrap();

    let mut w = Tensor::randn(&[2, 1], &rng).requires_grad_(true);
    let lr = 0.1f32;
    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for step in 0..200 {
        let pred = x.matmul(&w).unwrap();
        let loss = pred.sub(&y).unwrap().mul(&pred.sub(&y).unwrap()).unwrap().mean();
        w.zero_grad();
        loss.backward();
        let g = w.grad().unwrap().to_vec();
        let updated: Vec<f32> = w.to_vec().iter().zip(g).map(|(p, gg)| p - lr * gg).collect();
        w = Tensor::from_vec(updated, &[2, 1]).unwrap().requires_grad_(true);
        if step == 0 {
            first = loss.item();
        }
        last = loss.item();
    }
    assert!(last < first * 1e-3, "loss did not converge: {first} -> {last}");
    assert_close(w.to_vec()[0], 2.0, 1e-1, "w0");
    assert_close(w.to_vec()[1], -3.0, 1e-1, "w1");
}
