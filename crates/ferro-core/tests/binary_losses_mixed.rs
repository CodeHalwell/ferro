use ferro_core::{DType, Tensor};

fn f64_tensor(values: &[f64], shape: &[usize]) -> Tensor { Tensor::from_vec_f64(values.to_vec(), shape).unwrap() }

#[test]
fn wide_weights_are_not_narrowed_before_weighted_value_or_adjoint() {
    for shape in [&[][..], &[1][..]] {
        let x = Tensor::from_vec(vec![1e-30], &[1]).unwrap().requires_grad_(true).unwrap();
        let target = Tensor::zeros(&[1]).requires_grad_(true).unwrap();
        let weight = f64_tensor(&[1e40], shape);
        let loss = x.binary_loss(&target, None, Some(&weight), false, "none").unwrap();
        assert!((loss.to_vec()[0] - 1e10).abs() <= 1024.);
        loss.sum().backward();
        assert!((x.grad().unwrap().to_vec()[0] as f64 / 1e22 - 1.).abs() < 2e-6);
        assert!(target.grad().unwrap().to_vec()[0].is_infinite());
        assert!(weight.grad().is_none());
    }
    for scalar in [false, true] {
        let x = Tensor::from_vec(vec![-70.], &[1]).unwrap().requires_grad_(true).unwrap();
        let weight = f64_tensor(&[1e40], if scalar { &[] } else { &[1] });
        let loss = x.binary_loss(&Tensor::zeros(&[1]), None, Some(&weight), true, "none").unwrap();
        loss.sum().backward();
        if scalar {
            assert!(loss.to_vec()[0].is_nan());
            assert!(x.grad().unwrap().to_vec()[0].is_infinite());
        } else {
            assert_eq!(loss.to_vec(), vec![0.]);
            assert!((x.grad().unwrap().to_vec()[0] as f64 / 3975449600. - 1.).abs() < 2e-6);
        }
    }
}

#[test]
fn mixed_logits_keep_target_output_dtype_and_each_input_gradient_dtype() {
    for swapped in [false, true] {
        let (x, target) = if swapped {
            (f64_tensor(&[-0.7, 0., 1.2], &[3]), Tensor::from_vec(vec![0.2, 0.9, 0.5], &[3]).unwrap())
        } else {
            (Tensor::from_vec(vec![-0.7, 0., 1.2], &[3]).unwrap(), f64_tensor(&[0.2, 0.9, 0.5], &[3]))
        };
        let x = x.requires_grad_(true).unwrap();
        let target = target.requires_grad_(true).unwrap();
        let weight = Tensor::from_vec_i64(vec![1, 2, -1], &[3]).unwrap();
        let positive = Tensor::from_vec(vec![0.4, 2.3, 0.7], &[3]).unwrap();
        let loss = x.binary_loss(&target, Some(&positive), Some(&weight), true, "mean").unwrap();
        assert_eq!(loss.dtype(), target.dtype());
        loss.backward();
        assert_eq!(x.grad().unwrap().dtype(), x.dtype());
        assert_eq!(target.grad().unwrap().dtype(), target.dtype());
        assert!(weight.grad().is_none());
        assert!(positive.grad().is_none());
    }
}

#[test]
fn weighted_reduced_loss_gradients_match_finite_differences() {
    let x = Tensor::from_vec(vec![-0.7, 0., 1.2, -1.1], &[2, 2]).unwrap();
    let target = Tensor::from_vec(vec![0.2, 0.9, 0.5, 0.7], &[2, 2]).unwrap();
    let weight = f64_tensor(&[0.7, 1.3], &[2]);
    let positive = Tensor::from_vec(vec![0.4, 2.3], &[2, 1]).unwrap();
    ferro_core::testkit::grad_check_strict(&[x, target], |values| {
        values[0].binary_loss(&values[1], Some(&positive), Some(&weight), true, "mean").unwrap()
    });
}

#[test]
fn original_integer_and_float_weight_versions_reject_atomic_backward() {
    for selected in 0..4 {
        let x = Tensor::from_vec(vec![0.2, 0.7], &[2]).unwrap().requires_grad_(true).unwrap();
        let target = f64_tensor(&[0.3, 0.6], &[2]).requires_grad_(true).unwrap();
        let positive = Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap();
        let weight = f64_tensor(&[0.7, 1.3], &[2]);
        let loss = x.binary_loss(&target, Some(&positive), Some(&weight), true, "sum").unwrap();
        [&x, &target, &positive, &weight][selected]._bump_version_for_test();
        assert!(loss.grad_wrt(&[&x, &target], false).is_err());
        assert!(x.grad().is_none());
        assert!(target.grad().is_none());
    }
}

#[test]
fn empty_mixed_losses_and_invalid_metadata_remain_bounded() {
    for reduction in ["none", "sum", "mean"] {
        let x = Tensor::zeros(&[1usize << 60, 0]).requires_grad_(true).unwrap();
        let target = f64_tensor(&[], &[1usize << 60, 0]).requires_grad_(true).unwrap();
        let weight = Tensor::from_vec_i64(vec![2], &[]).unwrap();
        let out = x.binary_loss(&target, None, Some(&weight), true, reduction).unwrap();
        assert_eq!(out.dtype(), DType::F64);
        if reduction == "mean" { assert!(out.to_vec_f64()[0].is_nan()); }
        if reduction == "sum" { assert_eq!(out.to_vec_f64(), vec![0.]); }
        out.sum().backward();
        assert_eq!(x.grad().unwrap().shape(), x.shape());
        assert_eq!(target.grad().unwrap().shape(), target.shape());
    }
    let x = Tensor::from_vec(vec![0.3], &[1]).unwrap();
    assert!(x.binary_loss(&f64_tensor(&[0.2], &[1]), None, None, false, "none").is_err());
    assert!(x.binary_loss(&x, Some(&Tensor::from_vec_bool(vec![true], &[1]).unwrap()), None, true, "none").is_err());
    assert!(x.binary_loss(&x, None, Some(&Tensor::ones(&[2])), true, "none").is_err());
    assert!(x.binary_loss(&x, None, None, true, "bad").is_err());
    assert!(ferro_core::capture(|| x.binary_loss(&x, None, None, true, "none")).is_err());
    let tracked = x.requires_grad_(true).unwrap();
    assert!(tracked.binary_loss(&x, None, None, true, "mean").unwrap().grad_wrt(&[&tracked], true).is_err());
}
