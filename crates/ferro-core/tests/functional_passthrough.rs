use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use ferro_core::{capture, Error, Tensor};

#[test]
fn internal_first_order_vjp_preserves_only_unchanged_cotangents() {
    let x = Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::from_vec(vec![0.25, 0.75], &[2]).unwrap().requires_grad_(true).unwrap();
    let plus = x.add(&Tensor::ones(&[2])).unwrap();
    for output in [&x, &plus] {
        let result = output.vjp_wrt_preserve(&[&x], &seed).unwrap().remove(0);
        assert_eq!(result.id(), seed.id());
        assert!(result.requires_grad());
        let public = output.vjp_wrt(&[&x], &seed, false).unwrap().remove(0);
        assert_ne!(public.id(), seed.id());
        assert!(!public.requires_grad());
    }
    for output in [x.mul(&Tensor::full(&[2], 2.0)).unwrap(), x.add(&x).unwrap()] {
        let result = output.vjp_wrt_preserve(&[&x], &seed).unwrap().remove(0);
        assert!(!result.requires_grad());
        assert_eq!(result.to_vec(), vec![0.5, 1.5]);
        assert_ne!(result.id(), seed.id());
    }
    assert!(x.grad().is_none() && seed.grad().is_none());
    assert!(is_grad_enabled());
}

#[test]
fn passthrough_keeps_validation_mode_and_saved_version_contracts() {
    let base = Tensor::from_vec_f64(vec![2.0, 3.0], &[2]).unwrap();
    let x = base.requires_grad_(true).unwrap();
    let seed = Tensor::from_vec_f64(vec![0.25, 0.75], &[2]).unwrap().requires_grad_(true).unwrap();
    let output = x.add(&Tensor::from_vec_f64(vec![1.0; 2], &[2]).unwrap()).unwrap();
    with_grad_enabled(false, || {
        assert_eq!(output.vjp_wrt_preserve(&[&x], &seed).unwrap()[0].id(), seed.id());
        assert!(!is_grad_enabled());
        assert!(output.vjp_wrt_preserve(&[&x], &Tensor::ones(&[2])).is_err());
        assert!(!is_grad_enabled());
    });
    assert!(matches!(capture(|| output.vjp_wrt_preserve(&[&x], &seed)), Err(Error::Unsupported { .. })));
    base.copy_from(&Tensor::from_vec_f64(vec![5.0, 7.0], &[2]).unwrap()).unwrap();
    assert!(output.vjp_wrt_preserve(&[&x], &seed).is_err());
    assert!(x.grad().is_none() && seed.grad().is_none());
}

#[test]
fn watermark_is_nonallocating_and_orders_existing_and_new_nodes() {
    let before = Tensor::scalar(1.0);
    let watermark = ferro_core::tensor::tensor_id_watermark();
    assert!(before.id() < watermark);
    let after = Tensor::scalar(2.0);
    assert!(after.id() >= watermark);
    assert!(ferro_core::tensor::tensor_id_watermark() > after.id());
}

#[test]
fn private_sum_pullbacks_are_typed_views_while_public_results_remain_detached() {
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::ones(&[2, 3, 4]).to_dtype(dtype).requires_grad_(true).unwrap();
        for keepdim in [false, true] {
            let output = x.sum_dim(1, keepdim).unwrap();
            let base = Tensor::from_vec((1..=8).map(|i| i as f32).collect(), &[4, 2]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
            let seed = base.transpose(0, 1).unwrap();
            let seed = if keepdim { seed.unsqueeze(1).unwrap() } else { seed };
            let private = output.vjp_wrt_preserve(&[&x], &seed).unwrap().remove(0);
            assert!(private.requires_grad());
            assert!(private.is_leaf());
            assert_eq!(private.strides(), &[1, 0, 2]);
            assert_eq!(private.dtype(), dtype);
            let public = output.vjp_wrt(&[&x], &seed, false).unwrap().remove(0);
            assert!(!public.requires_grad());
            assert_eq!(private.to_vec_f64(), public.to_vec_f64());
            assert_eq!(private.sum().grad_wrt(&[&base], false).unwrap()[0].to_vec_f64(), vec![0.0; 8]);
            with_grad_enabled(false, || base.copy_leaf_from(&Tensor::full(&[4, 2], 9.0).to_dtype(dtype))).unwrap();
            assert_eq!(private.to_vec_f64(), vec![9.0; 24]);
            assert_ne!(private.to_vec_f64(), public.to_vec_f64());
        }
        let seed = Tensor::scalar(0.25).to_dtype(dtype).requires_grad_(true).unwrap();
        let output = x.sum();
        let result = output.vjp_wrt_preserve(&[&x], &seed).unwrap().remove(0);
        assert!(result.requires_grad());
        assert_eq!(result.strides(), &[0, 0, 0]);
        assert_eq!(result.to_vec_f64(), vec![0.25; 24]);
        assert!(!x.mean().vjp_wrt_preserve(&[&x], &seed).unwrap()[0].requires_grad());
        assert!(!output.vjp_wrt_preserve(&[&x], &seed.detach()).unwrap()[0].requires_grad());
        assert!(x.grad().is_none() && seed.grad().is_none());
        assert!(is_grad_enabled());
    }
}

#[test]
fn private_cat_and_stack_preserve_cotangent_views_and_duplicate_accumulation() {
    for dtype in [ferro_core::DType::F32, ferro_core::DType::F64] {
        let x = Tensor::ones(&[2, 1]).to_dtype(dtype).requires_grad_(true).unwrap();
        let y = Tensor::ones(&[2, 2]).to_dtype(dtype).requires_grad_(true).unwrap();
        let output = Tensor::cat(&[x.clone(), y.clone()], 1).unwrap();
        let base = Tensor::from_vec((1..=6).map(|i| i as f32).collect(), &[3, 2]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let seed = base.transpose(0, 1).unwrap();
        let parts = output.vjp_wrt_preserve(&[&x, &y], &seed).unwrap();
        for (part, offset) in parts.iter().zip([0, 2]) {
            assert!(part.requires_grad() && part.is_leaf());
            assert_eq!(part.strides(), &[1, 2]);
            assert_eq!(part.storage_offset(), offset);
            assert_eq!(part.sum().grad_wrt(&[&base], false).unwrap()[0].to_vec_f64(), vec![0.; 6]);
        }
        let public = output.vjp_wrt(&[&x, &y], &seed, false).unwrap();
        assert!(public.iter().all(|part| !part.requires_grad()));
        with_grad_enabled(false, || base.copy_leaf_from(&Tensor::full(&[3, 2], 9.).to_dtype(dtype))).unwrap();
        assert_eq!(parts[1].to_vec_f64(), vec![9.; 4]);
        assert_ne!(parts[1].to_vec_f64(), public[1].to_vec_f64());

        let stacked = Tensor::stack(&[y.clone(), y.clone()], 1).unwrap();
        let base = Tensor::from_vec((1..=8).map(|i| i as f32).collect(), &[2, 2, 2]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let seed = base.transpose(0, 2).unwrap();
        let duplicate = stacked.vjp_wrt_preserve(&[&y], &seed).unwrap().remove(0);
        assert!(!duplicate.requires_grad());
        assert_eq!(duplicate.to_vec_f64(), vec![4., 12., 6., 14.]);
        let second = Tensor::ones(&[2, 2]).to_dtype(dtype).requires_grad_(true).unwrap();
        let stacked = Tensor::stack(&[y.clone(), second.clone()], 1).unwrap();
        let parts = stacked.vjp_wrt_preserve(&[&y, &second], &seed).unwrap();
        for (part, offset) in parts.iter().zip([0, 2]) {
            assert!(part.requires_grad() && part.is_leaf());
            assert_eq!(part.strides(), &[1, 4]);
            assert_eq!(part.storage_offset(), offset);
        }
        if dtype == ferro_core::DType::F32 {
            let parts = stacked.vjp_wrt(&[&y, &second], &seed, true).unwrap();
            assert_eq!(parts[0].strides(), &[1, 4]);
            assert_eq!(parts[0].sum().grad_wrt(&[&base], false).unwrap()[0].to_vec(), vec![1., 1., 0., 0., 1., 1., 0., 0.]);
        }
        assert!(x.grad().is_none() && y.grad().is_none() && base.grad().is_none());
    }
}
