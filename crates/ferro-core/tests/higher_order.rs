use ferro_core::{Error, Tensor};

fn scalar(x: f32) -> Tensor { Tensor::scalar(x).requires_grad_(true).unwrap() }
fn tensor(v: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(v.to_vec(), shape).unwrap().requires_grad_(true).unwrap()
}
fn close(t: &Tensor, expected: f32) {
    assert!((t.item() - expected).abs() < 2e-5, "got {}, expected {expected}", t.item());
}
fn values(t: &Tensor, expected: &[f32]) {
    let actual = t.to_vec();
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) { assert!((a - e).abs() < 2e-5, "got {actual:?}, expected {expected:?}"); }
}
fn total(t: &Tensor) -> Tensor { t.sum() }

#[test]
fn scalar_sum_preserves_higher_order_graph_and_functional_state() {
    let x = tensor(&[1.0, -2.0, 3.0, -4.0], &[2, 2]);
    let y = x.mul(&x).unwrap().mul(&x).unwrap().sum();
    let g = y.grad_wrt(&[&x], true).unwrap().remove(0);
    values(&g, &[3.0, 12.0, 27.0, 48.0]);
    assert!(g.requires_grad());
    let gg = g.sum().grad_wrt(&[&x], true).unwrap().remove(0);
    values(&gg, &[6.0, -12.0, 18.0, -24.0]);
    values(&gg.sum().grad_wrt(&[&x], false).unwrap()[0], &[6.0; 4]);
    assert!(x.grad().is_none());
}

#[test]
fn scalar_mean_preserves_higher_order_graph_and_empty_semantics() {
    let x = tensor(&[1.0, -2.0, 3.0, -4.0], &[2, 2]);
    let z = x.transpose(0, 1).unwrap();
    let y = z.mul(&z).unwrap().mul(&z).unwrap().mean();
    let g = y.grad_wrt(&[&x], true).unwrap().remove(0);
    values(&g, &[0.75, 3.0, 6.75, 12.0]);
    assert!(g.requires_grad());
    let gg = g.sum().grad_wrt(&[&x], true).unwrap().remove(0);
    values(&gg, &[1.5, -3.0, 4.5, -6.0]);
    values(&gg.sum().grad_wrt(&[&x], false).unwrap()[0], &[1.5; 4]);
    assert!(x.grad().is_none());
    let empty = tensor(&[], &[0, 2]);
    assert!(empty.mean().item().is_nan());
    assert_eq!(empty.mean().grad_wrt(&[&empty], true).unwrap()[0].shape(), &[0, 2]);
    let s = scalar(2.0);
    close(&s.mean().grad_wrt(&[&s], true).unwrap()[0], 1.0);
}

#[test]
fn polynomial_second_derivative_and_functional_state() {
    let x = scalar(2.0);
    let y = x.mul(&x).unwrap().mul(&x).unwrap();
    let g = y.grad_wrt(&[&x], true).unwrap().remove(0);
    close(&g, 12.0);
    assert!(g.requires_grad());
    assert!(x.grad().is_none());
    close(&g.grad_wrt(&[&x], false).unwrap()[0], 12.0);
}

#[test]
fn mixed_partials_and_third_derivatives() {
    let x = scalar(1.5);
    let y = scalar(-0.7);
    let f = x.mul(&x).unwrap().mul(&y.mul(&y).unwrap()).unwrap();
    let g = f.grad_wrt(&[&x, &y], true).unwrap();
    let xy = g[0].grad_wrt(&[&y], true).unwrap().remove(0);
    let yx = g[1].grad_wrt(&[&x], true).unwrap().remove(0);
    close(&xy, 4.0 * 1.5 * -0.7);
    close(&yx, xy.item());
    close(&xy.grad_wrt(&[&x], false).unwrap()[0], 4.0 * -0.7);
}

#[test]
fn tanh_second_derivative_analytic() {
    for v in [-0.9_f32, 0.2, 1.1] {
        let x = scalar(v);
        let g = x.tanh().grad_wrt(&[&x], true).unwrap().remove(0);
        let t = v.tanh();
        close(&g, 1.0 - t*t);
        close(&g.grad_wrt(&[&x], false).unwrap()[0], -2.0*t*(1.0-t*t));
    }
}

#[test]
fn vector_seed_is_live_only_with_create_graph() {
    let x = tensor(&[2.0, 3.0], &[2]);
    let seed = tensor(&[0.5, -2.0], &[2]);
    let y = x.mul(&x).unwrap();
    let g = y.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
    values(&g, &[2.0, -12.0]);
    let next = total(&g).grad_wrt(&[&x, &seed], false).unwrap();
    values(&next[0], &[1.0, -4.0]);
    values(&next[1], &[4.0, 6.0]);
    assert!(!y.vjp_wrt(&[&x], &seed, false).unwrap()[0].requires_grad());
}

#[test]
fn unused_constants_duplicates_and_intermediate_selection() {
    let x = scalar(2.0);
    let unused = scalar(7.0);
    let c = Tensor::scalar(3.0);
    let a = x.mul(&x).unwrap();
    let y = a.mul(&x).unwrap();
    let g = y.grad_wrt(&[&a, &x, &unused, &x, &c], true).unwrap();
    for (t, expected) in g.iter().zip([2.0, 12.0, 0.0, 12.0, 0.0]) { close(t, expected); }
    close(&g[2].grad_wrt(&[&x], true).unwrap()[0], 0.0);
    let linear = x.mul(&c).unwrap().grad_wrt(&[&x], true).unwrap().remove(0);
    close(&linear, 3.0);
    close(&linear.grad_wrt(&[&x], true).unwrap()[0], 0.0);
    close(&x.grad_wrt(&[&x], false).unwrap()[0], 1.0);
}

#[test]
fn smooth_mlp_gradient_penalty_and_pde_parameter_gradients() {
    // u(x) = sum_j v_j tanh(x*w_j+b_j). Analytic mixed spatial/parameter derivatives.
    let xv = 0.4_f32;
    let ws = [0.7_f32, -1.1];
    let bs = [0.2_f32, 0.1];
    let vs = [0.6_f32, -0.3];
    let x = tensor(&[xv], &[1,1]);
    let w = tensor(&ws, &[1,2]);
    let b = tensor(&bs, &[2]);
    let v = tensor(&vs, &[2,1]);
    let u = x.matmul(&w).unwrap().add(&b).unwrap().tanh().matmul(&v).unwrap();
    let ux = u.grad_wrt(&[&x], true).unwrap().remove(0);
    let uxx = ux.grad_wrt(&[&x], true).unwrap().remove(0);
    let mut dx = 0.0;
    let mut dxx = 0.0;
    let mut dxw = vec![];
    let mut dxxw = vec![];
    for j in 0..2 {
        let t = (xv*ws[j]+bs[j]).tanh();
        let q = 1.0-t*t;
        dx += vs[j]*q*ws[j];
        dxx += -2.0*vs[j]*t*q*ws[j]*ws[j];
        dxw.push(vs[j]*(q-2.0*xv*ws[j]*t*q));
        dxxw.push(-2.0*vs[j]*(2.0*ws[j]*t*q+ws[j]*ws[j]*xv*q*(1.0-3.0*t*t)));
    }
    close(&ux, dx);
    close(&uxx, dxx);
    values(&ux.grad_wrt(&[&w], false).unwrap()[0], &dxw);
    values(&uxx.grad_wrt(&[&w], false).unwrap()[0], &dxxw);
    let penalty = ux.mul(&ux).unwrap();
    let gp = penalty.grad_wrt(&[&w], false).unwrap();
    values(&gp[0], &dxw.iter().map(|d| 2.0*dx*d).collect::<Vec<_>>());
    let residual = uxx.sub(&Tensor::scalar(0.25)).unwrap();
    let pde = residual.mul(&residual).unwrap();
    values(&pde.grad_wrt(&[&w], false).unwrap()[0], &dxxw.iter().map(|d| 2.0*(dxx-0.25)*d).collect::<Vec<_>>());
    penalty.backward();
    values(&w.grad().unwrap(), &gp[0].to_vec());
}

#[test]
fn broadcast_reduction_and_transpose_are_differentiable() {
    let x = tensor(&[0.2, -0.4, 0.7, 0.9, -0.5, 0.1], &[2,3]);
    let b = tensor(&[0.1, 0.2, 0.3], &[3]);
    let z = x.add(&b).unwrap().transpose(0, 1).unwrap();
    let f = total(&z.mul(&z).unwrap());
    let gb = f.grad_wrt(&[&b], true).unwrap().remove(0);
    values(&gb, &[2.6, -1.0, 2.8]);
    values(&total(&gb).grad_wrt(&[&x, &b], false).unwrap()[0], &[2.0; 6]);
    values(&total(&gb).grad_wrt(&[&b], false).unwrap()[0], &[4.0; 3]);
}

#[test]
fn smooth_exp_sigmoid_division_second_derivatives() {
    let x = scalar(0.7);
    let e = x.exp();
    close(&e.grad_wrt(&[&x], true).unwrap()[0].grad_wrt(&[&x], false).unwrap()[0], 0.7_f32.exp());
    let s = 1.0 / (1.0 + (-0.7_f32).exp());
    close(&x.sigmoid().grad_wrt(&[&x], true).unwrap()[0].grad_wrt(&[&x], false).unwrap()[0], s*(1.0-s)*(1.0-2.0*s));
    let inverse = Tensor::scalar(1.0).div(&x).unwrap();
    close(&inverse.grad_wrt(&[&x], true).unwrap()[0].grad_wrt(&[&x], false).unwrap()[0], 2.0 / 0.7_f32.powi(3));
}

#[test]
fn sqrt_norm_gradient_penalty_has_parameter_gradient() {
    let x = scalar(0.8);
    let w = scalar(1.2);
    let y = x.mul(&w).unwrap().tanh();
    let gx = y.grad_wrt(&[&x], true).unwrap().remove(0);
    let norm = gx.mul(&gx).unwrap().sqrt();
    let residual = norm.sub(&Tensor::scalar(1.0)).unwrap();
    let penalty = residual.mul(&residual).unwrap();
    let t = (0.8_f32*1.2).tanh();
    let q = 1.0-t*t;
    let expected = 2.0*(1.2*q-1.0)*q*(1.0-2.0*0.8*1.2*t);
    let gp = penalty.grad_wrt(&[&w], true).unwrap();
    close(&gp[0], expected);
    let z = scalar(1.7);
    let dz = z.sqrt().grad_wrt(&[&z], true).unwrap().remove(0);
    close(&dz.grad_wrt(&[&z], false).unwrap()[0], -0.25/1.7_f32.powf(1.5));
}

#[test]
fn changed_saved_constant_is_rejected() {
    let x = scalar(2.0);
    let c = Tensor::scalar(3.0);
    let y = x.mul(&c).unwrap();
    c.copy_from(&Tensor::scalar(4.0)).unwrap();
    assert!(matches!(y.grad_wrt(&[&x], true), Err(Error::Unsupported { .. })));
    assert!(matches!(y.grad_wrt(&[&x], false), Err(Error::Unsupported { .. })));
}

#[test]
fn rejects_unsupported_formulas_and_invalid_seeds_without_grad_side_effects() {
    let x = scalar(0.5);
    x.mul(&x).unwrap().backward();
    let before = x.grad().unwrap().item();
    assert!(matches!(x.relu().grad_wrt(&[&x], true), Err(Error::Unsupported { .. })));
    let custom = Tensor::scalar(1.0).record_fn(vec![x.clone()], |g| vec![g.detach_copy()]);
    assert!(matches!(custom.grad_wrt(&[&x], true), Err(Error::Unsupported { .. })));
    close(&custom.grad_wrt(&[&x], false).unwrap()[0], 1.0);
    let vector = tensor(&[1.0, 2.0], &[2]);
    assert!(matches!(vector.grad_wrt(&[&vector], true), Err(Error::InvalidShape { .. })));
    assert!(matches!(vector.vjp_wrt(&[&vector], &x, true), Err(Error::ShapeMismatch { .. })));
    let int_seed = Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap();
    assert!(matches!(vector.vjp_wrt(&[&vector], &int_seed, true), Err(Error::DtypeMismatch { .. })));
    close(&x.grad().unwrap(), before);
    // An unrelated unsupported branch is not on the selected derivative path.
    let y = scalar(0.8);
    close(&x.mul(&x).unwrap().add(&y.relu()).unwrap().grad_wrt(&[&x], true).unwrap()[0], 1.0);
}
