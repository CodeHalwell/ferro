use ferro_core::Tensor;

fn assert_allclose(a: &[f32], b: &[f32], tol: f32, what: &str) {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{what} elem {i}: {x} vs {y} (tol {tol})"
        );
    }
}

// VJP identity: for y = f(x), backward_with(v) must match backward() on
// y.mul(v).sum(), since d/dx <v, f(x)> = J^T v. The two paths associate
// float ops differently, so compare with a small tolerance, not bitwise.
#[test]
fn vjp_identity_elementwise_chain() {
    let xs = vec![0.7, -1.2, 2.3, -0.4, 1.1, -0.9];
    let ws = vec![1.3, 0.8, -0.5, 2.0, -1.1, 0.6];
    let vs = vec![0.3, -0.7, 1.1, 0.2, -0.4, 0.9];

    let x1 = Tensor::from_vec(xs.clone(), &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let w1 = Tensor::from_vec(ws.clone(), &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let v = Tensor::from_vec(vs.clone(), &[2, 3]).unwrap();
    x1.mul(&w1).unwrap().relu().backward_with(&v);

    let x2 = Tensor::from_vec(xs, &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let w2 = Tensor::from_vec(ws, &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    x2.mul(&w2)
        .unwrap()
        .relu()
        .mul(&v)
        .unwrap()
        .sum()
        .backward();

    assert_allclose(
        &x1.grad().unwrap().to_vec(),
        &x2.grad().unwrap().to_vec(),
        1e-4,
        "dx",
    );
    assert_allclose(
        &w1.grad().unwrap().to_vec(),
        &w2.grad().unwrap().to_vec(),
        1e-4,
        "dw",
    );
}

#[test]
fn vjp_identity_matmul() {
    let xs = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.5];
    let ws = vec![
        0.1, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, -0.8, -0.9, 1.0, -1.1, 1.2,
    ];
    let vs = vec![0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9];

    let x1 = Tensor::from_vec(xs.clone(), &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let w1 = Tensor::from_vec(ws.clone(), &[3, 4])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let v = Tensor::from_vec(vs.clone(), &[2, 4]).unwrap();
    x1.matmul(&w1).unwrap().backward_with(&v);

    let x2 = Tensor::from_vec(xs, &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let w2 = Tensor::from_vec(ws, &[3, 4])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    x2.matmul(&w2).unwrap().mul(&v).unwrap().sum().backward();

    assert_allclose(
        &x1.grad().unwrap().to_vec(),
        &x2.grad().unwrap().to_vec(),
        1e-4,
        "dx",
    );
    assert_allclose(
        &w1.grad().unwrap().to_vec(),
        &w2.grad().unwrap().to_vec(),
        1e-4,
        "dw",
    );
}

#[test]
fn vjp_identity_broadcast_add() {
    // b is [3] broadcast against [2, 3]; its backward must unbroadcast (sum
    // over the batch dim) under both entry points identically.
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bs = vec![0.1, -0.2, 0.3];
    let vs = vec![0.4, -0.1, 0.2, -0.3, 0.5, -0.6];

    let x1 = Tensor::from_vec(xs.clone(), &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let b1 = Tensor::from_vec(bs.clone(), &[3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let v = Tensor::from_vec(vs.clone(), &[2, 3]).unwrap();
    x1.add(&b1).unwrap().backward_with(&v);

    let x2 = Tensor::from_vec(xs, &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let b2 = Tensor::from_vec(bs, &[3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    x2.add(&b2).unwrap().mul(&v).unwrap().sum().backward();

    assert_allclose(
        &x1.grad().unwrap().to_vec(),
        &x2.grad().unwrap().to_vec(),
        1e-4,
        "dx",
    );
    assert_allclose(
        &b1.grad().unwrap().to_vec(),
        &b2.grad().unwrap().to_vec(),
        1e-4,
        "db",
    );
}

#[test]
fn repeated_backward_with_accumulates() {
    // Retain-graph parity with plain backward(): leaf grads accumulate across
    // calls. d/dx <v, x*x> = 2vx, so two calls give 4vx.
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let v = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let y = x.mul(&x).unwrap();
    y.backward_with(&v);
    y.backward_with(&v);
    assert_eq!(x.grad().unwrap().to_vec(), vec![4.0, 16.0, 36.0]);
}

#[test]
fn backward_equals_backward_with_ones() {
    // backward() is documented as the v=1 scalar case of backward_with(); on
    // the same graph the two must land on bitwise-identical leaf grads.
    let xs = vec![0.5, -1.0, 2.0, 0.3, -0.7, 1.5];
    let ws = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];

    let x1 = Tensor::from_vec(xs.clone(), &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let w1 = Tensor::from_vec(ws.clone(), &[3, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let loss1 = x1.matmul(&w1).unwrap().relu().sum();
    loss1.backward();

    let x2 = Tensor::from_vec(xs, &[2, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let w2 = Tensor::from_vec(ws, &[3, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let loss2 = x2.matmul(&w2).unwrap().relu().sum();
    let ones = Tensor::full_on(loss2.shape(), 1.0, loss2.device()).unwrap();
    loss2.backward_with(&ones);

    assert_eq!(x1.grad().unwrap().to_vec(), x2.grad().unwrap().to_vec());
    assert_eq!(w1.grad().unwrap().to_vec(), w2.grad().unwrap().to_vec());
}

#[test]
#[should_panic(expected = "does not match output shape")]
fn shape_mismatch_cotangent_panics() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let y = x.mul(&x).unwrap();
    let bad = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    y.backward_with(&bad);
}
