use ferro_core::testkit::grad_check;
use ferro_core::{Rng, Tensor};

fn assert_close(a: f32, b: f32, tol: f32, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
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
fn double_backward_accumulates_like_torch() {
    // Leaf grads accumulate across backward calls; interior grads are recomputed
    // fresh, so two backwards of x^2 give exactly 2 * 2x, not compounded junk.
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap().requires_grad_(true);
    let loss = x.mul(&x).unwrap().sum();
    loss.backward();
    loss.backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![4.0, 8.0, 12.0]);
}

#[test]
fn deep_graph_backward_and_drop() {
    // 100k chained ops: both the topological sort and graph teardown must be
    // iterative or this overflows the native stack.
    let x = Tensor::from_vec(vec![1.0], &[1]).unwrap().requires_grad_(true);
    let mut y = x.clone();
    for _ in 0..100_000 {
        y = y.add(&x).unwrap();
    }
    y.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![100_001.0]);
}

#[test]
#[should_panic(expected = "one gradient per input")]
fn record_fn_arity_mismatch_panics() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap().requires_grad_(true);
    let b = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap().requires_grad_(true);
    let out = Tensor::from_vec(vec![4.0, 6.0], &[2]).unwrap();
    let out = out.record_fn(vec![a.clone(), b.clone()], |g| vec![g.detach_copy()]);
    out.sum().backward();
}

#[test]
#[should_panic(expected = "does not match tensor shape")]
fn record_fn_wrong_grad_shape_panics() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap().requires_grad_(true);
    let out = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let out = out.record_fn(vec![a.clone()], |_| vec![Tensor::ones(&[3, 2])]);
    out.sum().backward();
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
