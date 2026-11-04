use ferro_core::{DType, Error, Tensor};

fn double(values: &[f64], shape: &[usize]) -> Tensor {
    Tensor::from_vec_f64(values.to_vec(), shape).unwrap()
}
fn leaf(values: &[f64], shape: &[usize]) -> Tensor { double(values, shape).requires_grad_(true).unwrap() }
fn close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() <= tol * (1.0 + b.abs()), "{a:?} != {b:?}"); }
}
fn check_grad(values: &[f64], shape: &[usize], f: impl Fn(&Tensor) -> Tensor) {
    let x = leaf(values, shape);
    f(&x).sum().backward();
    let actual = x.grad().unwrap();
    assert_eq!(actual.dtype(), DType::F64);
    let mut expected = Vec::new();
    for i in 0..values.len() {
        let mut plus = values.to_vec(); plus[i] += 1e-6;
        let mut minus = values.to_vec(); minus[i] -= 1e-6;
        expected.push((f(&double(&plus, shape)).sum().to_vec_f64()[0] - f(&double(&minus, shape)).sum().to_vec_f64()[0]) / 2e-6);
    }
    close(&actual.to_vec_f64(), &expected, 2e-8);
}

#[test]
fn float64_values_and_gradients_never_narrow() {
    let values = [1.0 + 2f64.powi(-40), 1e100, 1e-300, f64::from_bits(7)];
    let x = leaf(&values, &[4]);
    let y = x.mul(&double(&[2.0], &[])).unwrap();
    assert_eq!(y.to_vec_f64(), values.map(|v| v * 2.0));
    y.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec_f64(), vec![2.0; 4]);
    let held = x.grad().unwrap();
    y.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec_f64(), vec![4.0; 4]);
    assert_eq!(held.id(), x.grad().unwrap().id());
    assert_eq!(held.to_vec_f64(), vec![4.0; 4]);
    let small = leaf(&[1e-300], &[]);
    let denominator = leaf(&[1e-300], &[]);
    small.div(&denominator).unwrap().backward();
    close(&small.grad().unwrap().to_vec_f64(), &[1e300], 1e-15);
    close(&denominator.grad().unwrap().to_vec_f64(), &[-1e300], 1e-15);
}

#[test]
fn broadcast_arithmetic_and_unary_adjoints() {
    let values = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4];
    check_grad(&values, &[2, 3], |x| {
        let bias = double(&[0.7, 1.1, 1.5], &[3]);
        x.mul(&bias).unwrap().add(&x.div(&bias).unwrap()).unwrap().sub(x).unwrap().tanh()
    });
    let unary: Vec<Box<dyn Fn(&Tensor) -> Tensor>> = vec![
        Box::new(|x| x.neg()), Box::new(|x| x.exp()), Box::new(|x| x.sigmoid()),
        Box::new(|x| x.log()), Box::new(|x| x.sqrt()), Box::new(|x| x.abs()),
        Box::new(|x| x.powf64(1.23456789012345).unwrap()), Box::new(|x| x.square().unwrap()),
        Box::new(|x| x.clamp_f64(0.6, 2.2).unwrap()), Box::new(|x| x.relu()),
    ];
    for f in unary { check_grad(&values, &[2, 3], f); }
    check_grad(&[0.7, 1.1, 1.5], &[3], |b| double(&values, &[2, 3]).mul(b).unwrap().mean_dim(0, false).unwrap());
}

#[test]
fn matrices_batches_vectors_and_strides_have_f64_adjoints() {
    let av = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
    let bv = [0.3, 0.7, 1.1, 1.5, 1.9, 2.3];
    check_grad(&av, &[3, 2], |a| a.transpose(0, 1).unwrap().matmul(&double(&bv, &[3, 2])).unwrap());
    check_grad(&bv, &[3, 2], |b| double(&av, &[2, 3]).matmul(b).unwrap());
    check_grad(&av, &[2, 1, 3], |a| a.bmm(&double(&bv, &[2, 3, 1])).unwrap());
    check_grad(&bv, &[2, 3, 1], |b| double(&av, &[2, 1, 3]).bmm(b).unwrap());
    check_grad(&av, &[2, 1, 3], |a| a.matmul(&double(&bv, &[3, 2])).unwrap());
    check_grad(&bv, &[3, 2], |b| double(&av, &[2, 1, 3]).matmul(b).unwrap());
    check_grad(&av[..3], &[3], |a| a.matmul(&double(&bv, &[3, 2])).unwrap());
    check_grad(&bv[..3], &[3], |b| double(&av, &[2, 3]).matmul(b).unwrap());
    let x = leaf(&[1e100, 2e100], &[2]);
    let y = x.matmul(&double(&[1e-100, 2e-100], &[2])).unwrap();
    assert_eq!(y.shape(), &[]);
    close(&y.to_vec_f64(), &[5.0], 1e-15);
}

#[test]
fn typed_views_and_selectors_preserve_history() {
    let values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6];
    check_grad(&values, &[2, 4], |x| {
        let y = x.slice(1, 1, 4, 2).unwrap().transpose(0, 1).unwrap().reshape(&[4]).unwrap();
        Tensor::cat(&[y.clone(), y], 0).unwrap().index_select(0, &[1, 1, 6]).unwrap()
    });
    check_grad(&values, &[2, 4], |x| x.gather(1, &Tensor::from_vec_i64(vec![3, 0, 0, 2], &[2, 2]).unwrap()).unwrap());
    check_grad(&values[..2], &[2, 1], |x| {
        let expanded = x.expand(&[2, 3]).unwrap();
        let mask = Tensor::from_vec_bool(vec![true, false, true], &[3]).unwrap();
        Tensor::where_cond(&mask, &expanded, &double(&[0.5], &[])).unwrap()
    });
    let x = leaf(&[2.0, 2.0, 1.0], &[3]);
    x.max().unwrap().backward();
    close(&x.grad().unwrap().to_vec_f64(), &[0.5, 0.5, 0.0], 0.0);
}

#[test]
fn empty_outputs_and_reduction_capacity_are_checked() {
    let x = leaf(&[], &[2, 0, 3]);
    assert_eq!(x.sum().to_vec_f64(), vec![0.0]);
    assert!(x.mean().to_vec_f64()[0].is_nan());
    assert_eq!(x.sum_dim(1, false).unwrap().to_vec_f64(), vec![0.0; 6]);
    assert!(x.mean_dim(1, false).unwrap().to_vec_f64().iter().all(|x| x.is_nan()));
    x.sum().backward();
    assert_eq!(x.grad().unwrap().shape(), &[2, 0, 3]);
    let huge_empty = double(&[], &[usize::MAX, 0]);
    assert!(matches!(huge_empty.sum_dim(1, false), Err(Error::InvalidShape { .. })));
    assert!(matches!(huge_empty.matmul(&double(&[], &[0, 2])), Err(Error::InvalidShape { .. })));
    let a = leaf(&[], &[2, 0]);
    let b = leaf(&[], &[0, 3]);
    let y = a.matmul(&b).unwrap();
    assert_eq!(y.to_vec_f64(), vec![0.0; 6]);
    y.sum().backward();
    assert!(a.grad().unwrap().to_vec_f64().is_empty());
    assert!(b.grad().unwrap().to_vec_f64().is_empty());
    let a = leaf(&[1.0, 2.0], &[1, 1, 2]);
    let b = leaf(&[], &[1 << (usize::BITS - 3), 2, 0]);
    let y = a.matmul(&b).unwrap();
    assert_eq!(y.shape(), &[1 << (usize::BITS - 3), 1, 0]);
    y.sum().backward();
    assert_eq!(a.grad().unwrap().to_vec_f64(), vec![0.0, 0.0]);
    assert!(b.grad().unwrap().to_vec_f64().is_empty());
}

#[test]
fn functional_first_order_is_typed_and_has_no_persistent_grad_side_effects() {
    let x = leaf(&[0.3, 0.7], &[2]);
    let unused = leaf(&[0.2], &[]);
    let y = x.square().unwrap();
    assert!(x.depends_on(&x));
    assert!(y.depends_on(&x));
    assert!(!y.depends_on(&unused));
    assert!(!y.detach_copy().depends_on(&x));
    assert!(x.mul(&double(&[0.0], &[])).unwrap().depends_on(&x));
    assert!(y.has_grad_leaf_except(&[]));
    assert!(!y.has_grad_leaf_except(&[&x]));
    assert!(!y.has_grad_leaf_except(&[&y]));
    assert!(y.add(&unused).unwrap().has_grad_leaf_except(&[&x]));
    assert!(!y.detach_copy().has_grad_leaf_except(&[]));
    let seed = leaf(&[0.2, 0.4], &[2]);
    y.sum().backward();
    let before = x.grad().unwrap();
    let grads = y.vjp_wrt(&[&x, &unused], &seed, false).unwrap();
    close(&grads[0].to_vec_f64(), &[0.12, 0.56], 1e-15);
    assert_eq!(grads[1].dtype(), DType::F64);
    assert_eq!(grads[1].to_vec_f64(), vec![0.0]);
    assert!(!grads[0].requires_grad());
    assert!(seed.grad().is_none());
    assert_eq!(x.grad().unwrap().to_vec_f64(), before.to_vec_f64());
    assert!(matches!(y.vjp_wrt(&[&x], &seed, true), Err(Error::DtypeMismatch { .. })));
    assert!(matches!(y.vjp_wrt(&[&x], &Tensor::ones(&[2]), false), Err(Error::DtypeMismatch { .. })));
    let scalar = y.sum();
    close(&scalar.grad_wrt(&[&x], false).unwrap()[0].to_vec_f64(), &[0.6, 1.4], 1e-15);
}

#[test]
fn mismatched_contributions_and_stale_parents_are_rejected() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let x = leaf(&[2.0], &[]);
    let bad = double(&[3.0], &[]).record_fn(vec![x.clone()], |_| vec![Tensor::scalar(1.0)]);
    assert!(matches!(bad.grad_wrt(&[&x], false), Err(Error::DtypeMismatch { expected: DType::F64, got: DType::F32, .. })));
    assert!(x.grad().is_none());
    assert!(catch_unwind(AssertUnwindSafe(|| bad.backward())).is_err());
    assert!(x.grad().is_none());
    let original = Tensor::scalar(2.0);
    let parent = original.requires_grad_(true).unwrap();
    let mixed = double(&[3.0], &[]).record_fn(vec![parent.clone()], |_| vec![Tensor::scalar(1.0)]);
    original.copy_from(&Tensor::scalar(4.0)).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| mixed.backward())).is_err());
    assert!(parent.grad().is_none());
}

#[test]
fn integer_arithmetic_is_exact_wrapping_and_nondifferentiable() {
    let a = Tensor::from_vec_i64(vec![i64::MAX, i64::MIN, (1 << 53) + 1], &[3]).unwrap();
    let one = Tensor::from_vec_i64(vec![1], &[]).unwrap();
    assert_eq!(a.add(&one).unwrap().to_vec_i64(), vec![i64::MIN, i64::MIN + 1, (1 << 53) + 2]);
    assert_eq!(a.neg().to_vec_i64(), vec![-i64::MAX, i64::MIN, -((1 << 53) + 1)]);
    assert_eq!(a.mul(&a).unwrap().to_vec_i64(), a.to_vec_i64().iter().map(|x| x.wrapping_mul(*x)).collect::<Vec<_>>());
    assert_eq!(a.sum().to_vec_i64(), vec![1 << 53]);
    assert_eq!(a.sum_dim(0, true).unwrap().shape(), &[1]);
    assert_eq!(a.max().unwrap().to_vec_i64(), vec![i64::MAX]);
    assert!(!a.add(&one).unwrap().requires_grad());
    assert_eq!(a.div(&one).unwrap().dtype(), DType::F32);
    assert!(a.requires_grad_(true).is_err());
}

#[test]
fn invalid_typed_requests_reject_before_computation() {
    let x = leaf(&[1.0, 2.0], &[2]);
    assert_eq!(x.add(&Tensor::ones(&[2])).unwrap().dtype(), DType::F64);
    assert_eq!(Tensor::ones(&[2]).add(&x).unwrap().dtype(), DType::F64);
    assert!(x.matmul(&double(&[1.0], &[])).is_err());
    assert!(x.bmm(&x).is_err());
    assert!(ferro_core::capture(|| x.add(&x)).is_err());
    assert!(ferro_core::capture(|| x.sum_dim(0, false)).is_err());
    assert!(!ferro_core::capture::is_recording());
    assert!(x.grad().is_none());
    x.sum().backward();
    assert_eq!(x.grad().unwrap().dtype(), DType::F64);
    let oversized = double(&[1.0], &[]).expand(&[usize::MAX]).unwrap();
    assert!(matches!(oversized.powf64(2.0), Err(Error::InvalidShape { .. })));
}

#[test]
fn powers_zero_and_one_preserve_boundary_derivatives() {
    for p in [0.0, 1.0] {
        let expected = vec![p; 4];
        let x = leaf(&[0.0, -0.0, f64::INFINITY, f64::NAN], &[4]);
        x.powf64(p).unwrap().sum().backward();
        assert_eq!(x.grad().unwrap().to_vec_f64(), expected);
        let x = Tensor::from_vec(vec![0.0, -0.0, f32::INFINITY, f32::NAN], &[4]).unwrap().requires_grad_(true).unwrap();
        x.powf(p as f32).sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![p as f32; 4]);
    }
}

#[test]
fn typed_materialization_checks_precede_reading_broadcast_views() {
    let huge = double(&[1.0], &[]).expand(&[isize::MAX as usize / 8 + 1]).unwrap();
    assert!(matches!(huge.index_select(0, &[]), Err(Error::InvalidShape { .. })));
    assert!(matches!(huge.gather(0, &Tensor::from_vec_i64(vec![], &[0]).unwrap()), Err(Error::InvalidShape { .. })));
    assert!(matches!(Tensor::cat(&[huge.clone()], 0), Err(Error::InvalidShape { .. })));
    let same_shape = huge.reshape(huge.shape()).unwrap();
    assert_eq!(same_shape.strides(), huge.strides());
    assert_eq!(same_shape._storage_ptr(), huge._storage_ptr());
    assert!(matches!(ferro_core::capture(|| leaf(&[1.0], &[1]).index_select(0, &[])), Err(Error::Unsupported { .. })));
}
