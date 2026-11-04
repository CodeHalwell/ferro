use ferro_core::{DType, Error, Tensor};

fn double(values: &[f64], shape: &[usize]) -> Tensor { Tensor::from_vec_f64(values.to_vec(), shape).unwrap() }
fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() <= tolerance * (1.0 + b.abs()), "{a} != {b}"); }
}

#[test]
fn dimensional_dtype_precedence_matches_tensor_promotion() {
    let floats = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let doubles = double(&[1.0, 2.0], &[2]);
    let fs = Tensor::scalar(0.5);
    let ds = double(&[0.5], &[]);
    for (a, b, dtype) in [(&floats, &doubles, DType::F64), (&floats, &ds, DType::F32),
                          (&ds, &floats, DType::F32), (&fs, &doubles, DType::F64), (&fs, &ds, DType::F64)] {
        assert_eq!(a.add(b).unwrap().dtype(), dtype);
        assert_eq!(a.mul(b).unwrap().dtype(), dtype);
        assert_eq!(a.div(b).unwrap().dtype(), dtype);
    }
    let ints = Tensor::from_vec_i64(vec![(1 << 53) + 1, 3], &[2]).unwrap();
    assert_eq!(ints.add(&ds).unwrap().dtype(), DType::F64);
    assert_eq!(ints.add(&fs).unwrap().dtype(), DType::F32);
    assert_eq!(ints.div(&ints).unwrap().dtype(), DType::F32);
    assert_eq!(ints.div(&ints).unwrap().to_vec(), vec![1.0, 1.0]);
    assert!(matches!(floats.reshape(&[1, 2]).unwrap().matmul(&doubles.reshape(&[2, 1]).unwrap()), Err(Error::DtypeMismatch { .. })));
}

#[test]
fn mixed_gradients_restore_each_original_floating_dtype() {
    let x = Tensor::from_vec(vec![0.25, 0.5], &[2, 1]).unwrap().requires_grad_(true).unwrap();
    let y = double(&[1.0 + 2f64.powi(-40), 0.75], &[1, 2]).requires_grad_(true).unwrap();
    let loss = x.mul(&y).unwrap().add(&x.div(&y).unwrap()).unwrap().sum();
    assert_eq!(loss.dtype(), DType::F64);
    let grads = loss.grad_wrt(&[&x, &y], false).unwrap();
    assert!(x.grad().is_none() && y.grad().is_none());
    assert_eq!(grads[0].dtype(), DType::F32);
    assert_eq!(grads[1].dtype(), DType::F64);
    loss.backward();
    assert_eq!(x.grad().unwrap().to_vec(), grads[0].to_vec());
    assert_eq!(y.grad().unwrap().to_vec_f64(), grads[1].to_vec_f64());
    let values = y.to_vec_f64();
    close(&grads[0].to_vec_f64(), &[values.iter().map(|v| v + 1.0 / v).sum(); 2], 1e-7);
    close(&grads[1].to_vec_f64(), &values.iter().map(|v| 0.75 * (1.0 - 1.0 / (v * v))).collect::<Vec<_>>(), 1e-12);
    let scalar = double(&[1.0000000000001], &[]).requires_grad_(true).unwrap();
    let out = x.mul(&scalar).unwrap();
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.sum().grad_wrt(&[&scalar], false).unwrap()[0].dtype(), DType::F64);
    assert_eq!(out.sum().grad_wrt(&[&scalar], false).unwrap()[0].to_vec_f64(), vec![0.75]);
}

#[test]
fn differentiable_casts_preserve_identity_or_owned_copy_and_history() {
    let original = Tensor::from_vec(vec![0.2, 0.4], &[2]).unwrap();
    let x = original.requires_grad_(true).unwrap();
    assert_eq!(x.to_dtype_autograd(DType::F32, false).unwrap().id(), x.id());
    let copy = x.to_dtype_autograd(DType::F32, true).unwrap();
    assert_ne!(copy.id(), x.id());
    let d = x.to_dtype_autograd(DType::F64, false).unwrap();
    assert!(d.requires_grad() && d.depends_on(&x));
    d.square().unwrap().sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![0.4, 0.8]);
    assert!(!x.to_dtype(DType::F64).requires_grad(), "legacy explicit detached cast stays detached");
    assert!(!d.to_dtype_autograd(DType::I64, false).unwrap().requires_grad());
    original.copy_from(&Tensor::zeros(&[2])).unwrap();
    assert_eq!(copy.to_vec(), vec![0.2, 0.4]);
    assert_eq!(d.to_vec_f64(), vec![0.2f32 as f64, 0.4f32 as f64]);
    assert_eq!(d.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec(), vec![1.; 2], "cast adjoints use dtype and shape, not source values");
    let nonlinear = d.square().unwrap().sum();
    d.detach().copy_from(&Tensor::from_vec_f64(vec![9.; 2], &[2]).unwrap()).unwrap();
    assert!(nonlinear.grad_wrt(&[&x], false).is_err(), "nonlinear consumers still save the cast output");
}

#[test]
fn boolean_arithmetic_has_explicit_torch_rules() {
    let a = Tensor::from_vec_bool(vec![true, false, true], &[3]).unwrap();
    let b = Tensor::from_vec_bool(vec![true, true, false], &[3]).unwrap();
    assert_eq!(a.add(&b).unwrap().to_vec_bool(), vec![true, true, true]);
    assert_eq!(a.mul(&b).unwrap().to_vec_bool(), vec![true, false, false]);
    assert!(a.sub(&b).is_err());
    assert!(Tensor::ones(&[3]).sub(&a).is_err());
    let integers = Tensor::from_vec_i64(vec![2, 3, 4], &[3]).unwrap();
    assert_eq!(a.add(&integers).unwrap().to_vec_i64(), vec![3, 3, 5]);
    assert_eq!(a.div(&integers).unwrap().to_vec(), vec![0.5, 0.0, 0.25]);
    assert!(!a.add(&integers).unwrap().requires_grad());
}

#[test]
fn casts_and_promotions_reject_capture_and_nonportable_integer_values() {
    let x = Tensor::ones(&[2]);
    let y = double(&[1.0, 2.0], &[2]);
    assert!(ferro_core::capture(|| x.add(&y)).is_err());
    assert!(ferro_core::capture(|| x.to_dtype_autograd(DType::F64, false)).is_err());
    for value in [f64::NAN, f64::INFINITY, i64::MAX as f64] {
        assert!(double(&[value], &[]).to_dtype_autograd(DType::I64, false).is_err());
    }
    assert_eq!(double(&[-1.9, 2.9], &[2]).to_dtype_autograd(DType::I64, false).unwrap().to_vec_i64(), vec![-1, 2]);
}

#[test]
fn dropout_preserves_double_scale_gradients_and_counter_replay() {
    let p = 0.1234567890123456;
    let x = double(&vec![1.0; 256], &[256]).requires_grad_(true).unwrap();
    let y = x.dropout_f64(p, true, 17, 9).unwrap();
    let mask = y.to_vec_f64();
    let scale = 1.0 / (1.0 - p);
    assert_ne!(scale, (scale as f32) as f64);
    assert!(mask.contains(&0.0) && mask.contains(&scale));
    assert!(mask.iter().all(|&v| v == 0.0 || v == scale));
    assert_eq!(mask, x.dropout_f64(p, true, 17, 9).unwrap().to_vec_f64());
    assert_ne!(mask, x.dropout_f64(p, true, 17, 10).unwrap().to_vec_f64());
    y.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec_f64(), mask);
    assert_eq!(x.dropout_f64(p, false, 17, 9).unwrap().id(), x.id());
    assert_eq!(x.dropout_f64(0.0, true, 17, 9).unwrap().id(), x.id());
    assert_eq!(x.dropout_f64(1.0, true, 17, 9).unwrap().to_vec_f64(), vec![0.0; 256]);
    assert!(x.dropout_f64(f64::NAN, true, 17, 9).is_err());
    assert!(ferro_core::capture(|| x.dropout_f64(p, true, 17, 9)).is_err());
}

#[test]
fn mixed_comparisons_use_dimensioned_dtype_precedence_without_gradients() {
    let ints = Tensor::from_vec_i64(vec![(1 << 24) + 1, (1 << 53) + 1], &[2]).unwrap();
    let single = Tensor::scalar((1 << 24) as f32);
    let double_scalar = double(&[(1 << 24) as f64], &[]);
    assert_eq!(ints.eq_tensor(&single).unwrap().to_vec_bool(), vec![true, false]);
    assert_eq!(ints.eq_tensor(&double_scalar).unwrap().to_vec_bool(), vec![false, false]);
    let x = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    let ds = double(&[1.0 + 2f64.powi(-40)], &[]).requires_grad_(true).unwrap();
    assert_eq!(x.eq_tensor(&ds).unwrap().to_vec_bool(), vec![true, true]);
    assert!(!x.eq_tensor(&ds).unwrap().requires_grad());
    assert_eq!(x.eq_tensor(&ds.reshape(&[1]).unwrap()).unwrap().to_vec_bool(), vec![false, false]);
    let empty = double(&[], &[0, 2]);
    assert_eq!(empty.lt_tensor(&ints).unwrap().shape(), &[0, 2]);
    let huge = double(&[1.0], &[1]).expand(&[isize::MAX as usize / 8 + 1]).unwrap();
    assert!(matches!(huge.eq_tensor(&double_scalar), Err(Error::InvalidShape { .. })));
    assert!(ferro_core::capture(|| ints.eq_tensor(&double_scalar)).is_err());
}

#[test]
fn typed_batched_triangles_preserve_values_and_f64_adjoint() {
    let values = [1e100, 1.0 + 2f64.powi(-40), 1e-200, -3.0, 4.0, 5.0, 6.0, 7.0];
    let x = double(&values, &[2, 2, 2]).requires_grad_(true).unwrap();
    let upper = x.triu(0).unwrap();
    assert_eq!(upper.dtype(), DType::F64);
    assert_eq!(upper.to_vec_f64(), vec![values[0], values[1], 0.0, -3.0, 4.0, 5.0, 0.0, 7.0]);
    let seed = double(&[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], &[2, 2, 2]);
    upper.mul(&seed).unwrap().sum().backward();
    assert_eq!(x.grad().unwrap().to_vec_f64(), vec![0.25, 0.5, 0.0, 1.0, 1.25, 1.5, 0.0, 2.0]);
    let ints = Tensor::from_vec_i64(vec![i64::MIN, i64::MAX, (1 << 53) + 1, -7], &[2, 2]).unwrap();
    assert_eq!(ints.tril(0).unwrap().to_vec_i64(), vec![i64::MIN, 0, (1 << 53) + 1, -7]);
    assert_eq!(ints.triu(i64::MIN).unwrap().to_vec_i64(), ints.to_vec_i64());
    assert_eq!(ints.tril(i64::MAX).unwrap().to_vec_i64(), ints.to_vec_i64());
    for shape in [vec![2, 0, 3], vec![0, 2, 3], vec![2, 3, 0]] {
        assert_eq!(double(&[], &shape).triu(0).unwrap().shape(), shape);
        assert_eq!(double(&[], &shape).tril(0).unwrap().shape(), shape);
    }
    let bools = Tensor::from_vec_bool(vec![true; 8], &[2, 2, 2]).unwrap();
    assert_eq!(bools.triu(0).unwrap().to_vec_bool(), vec![true, true, false, true, true, true, false, true]);
    assert!(ferro_core::capture(|| x.triu(0)).is_err());
}
