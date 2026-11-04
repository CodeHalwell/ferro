use ferro_core::{DType, Tensor, autograd::with_grad_enabled, forward_ad::DualLevel};

fn transform(input: &Tensor, operation: usize) -> Tensor {
    match operation { 0 => input.floor(), 1 => input.ceil(), 2 => input.trunc(), 3 => input.round(),
        4 => input.frac(), 5 => input.rsqrt(), _ => unreachable!() }.unwrap()
}

#[test]
fn typed_rounding_preserves_exact_values_and_rejects_unsupported_storage() {
    let input = Tensor::from_vec_f64(vec![1e300, -1e300, 0.5, 1.5, 2.5, -0.0, -1e-300, 1.0000000000000002], &[2, 4]).unwrap();
    let expected = [1e300, -1e300, 0.0, 2.0, 2.0, -0.0, -0.0, 1.0];
    assert_eq!(input.round().unwrap().to_vec_f64().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v: &f64| v.to_bits()).collect::<Vec<_>>());
    let strided = input.transpose(0, 1).unwrap();
    assert_eq!(strided.floor().unwrap().strides(), strided.strides());
    let gapped = Tensor::from_vec((0..24).map(|v| v as f32).collect(), &[4, 6]).unwrap().slice(0, 0, 4, 2).unwrap().slice(1, 0, 6, 2).unwrap().transpose(0, 1).unwrap();
    assert_eq!(gapped.floor().unwrap().strides(), &[1, 3]);
    assert_eq!(strided.ceil().unwrap().to_vec_f64(), strided.to_vec_f64().into_iter().map(f64::ceil).collect::<Vec<_>>());
    let integer = Tensor::from_vec_i64(vec![i64::MIN, i64::MAX, (1 << 54) + 1], &[3]).unwrap();
    for operation in 0..4 {
        let output = transform(&integer, operation);
        assert_eq!(output.dtype(), DType::I64);
        assert_eq!(output.to_vec_i64(), integer.to_vec_i64());
        assert_ne!(output._storage_ptr(), integer._storage_ptr());
    }
    assert!(integer.round_decimals(0).is_err());
    assert!(integer.frac().is_err());
    let boolean = Tensor::from_vec_bool(vec![false, true], &[2]).unwrap();
    assert!(boolean.floor().is_err() && boolean.round().is_err() && boolean.frac().is_err());
    assert_eq!(boolean.rsqrt().unwrap().to_vec(), vec![f32::INFINITY, 1.0]);
}

#[test]
fn decimal_rounding_uses_typed_stages_and_keeps_signed_zero() {
    for dtype in [DType::F32, DType::F64] {
        let x = Tensor::from_vec_f64(vec![-0.0, -0.25, 1.25, 123.45], &[4]).unwrap().to_dtype(dtype);
        let out = x.round_decimals(1).unwrap().to_vec_f64();
        let expected = if dtype == DType::F32 { vec![-0.0, -0.2f32 as f64, 1.2f32 as f64, 123.4f32 as f64] } else { vec![-0.0, -0.2, 1.2, 123.4] };
        assert_eq!(out, expected);
        assert!(out[0].is_sign_negative());
        for decimals in [i64::MIN, -1000, 1000, i64::MAX] {
            assert!(x.round_decimals(decimals).unwrap().to_vec_f64().iter().all(|v| v.is_nan()));
        }
    }
}

#[test]
fn zero_and_identity_adjoints_do_not_read_mutated_inputs_or_nan_cotangents() {
    for dtype in [DType::F32, DType::F64] { for operation in 0..5 {
        let x = Tensor::from_vec_f64(vec![0.5, 1.5, f64::INFINITY, f64::NAN], &[4]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let output = transform(&x, operation);
        let seed = Tensor::from_vec_f64(vec![1.0, -0.0, f64::INFINITY, f64::NAN], &[4]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        with_grad_enabled(false, || x.copy_leaf_from(&Tensor::zeros(x.shape()).to_dtype(dtype))).unwrap();
        output.detach().copy_from(&Tensor::zeros(output.shape()).to_dtype(dtype)).unwrap();
        let gradient = output.vjp_wrt_preserve(&[&x], &seed).unwrap().remove(0);
        if operation == 4 { assert_eq!(gradient.id(), seed.id()); }
        else { assert!(!gradient.requires_grad()); assert_eq!(gradient.to_vec_f64(), vec![0.0; 4]); }
        assert!(x.grad().is_none());
    } }
    let x = Tensor::ones(&[3]).requires_grad_(true).unwrap();
    let seed = Tensor::ones(&[3]).requires_grad_(true).unwrap();
    let gradient = x.frac().unwrap().vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
    assert_eq!(gradient.id(), seed.id());
    assert_eq!(gradient.sum().grad_wrt(&[&seed], false).unwrap()[0].to_vec(), vec![1.0; 3]);
}

#[test]
fn reciprocal_square_root_saves_only_its_output_and_matches_signed_zero_adjoint() {
    for dtype in [DType::F32, DType::F64] {
        let x = Tensor::from_vec_f64(vec![0.0, -0.0, 0.5, 1.5, f64::INFINITY], &[5]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let output = x.rsqrt().unwrap();
        let y = output.to_vec_f64();
        with_grad_enabled(false, || x.copy_leaf_from(&Tensor::ones(x.shape()).to_dtype(dtype))).unwrap();
        let gradient = output.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec_f64();
        let expected: Vec<f64> = if dtype == DType::F32 { y.iter().map(|y| (-0.5f32 * (*y as f32).powi(3)) as f64).collect() }
            else { y.iter().map(|y| -0.5 * y.powi(3)).collect() };
        assert_eq!(gradient.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        output.detach().copy_from(&Tensor::ones(output.shape()).to_dtype(dtype)).unwrap();
        assert!(output.sum().grad_wrt(&[&x], false).is_err());
        assert!(x.grad().is_none());
    }
}

#[test]
fn direct_tangents_keep_f64_precision_zero_rules_and_live_seed_identity() {
    for dtype in [DType::F32, DType::F64] { for operation in 0..6 {
        let x = Tensor::from_vec(vec![0.5, 1.5], &[2]).unwrap().requires_grad_(true).unwrap();
        let seed = Tensor::from_vec_f64(vec![1.0000000000000002, 2.0], &[2]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let level = DualLevel::new(&[&x], &[&seed]).unwrap();
        let output = transform(&x, operation);
        let tangent = level.tangent(&output).unwrap().unwrap();
        drop(level);
        if operation < 4 { assert_eq!(tangent.to_vec_f64(), vec![0.0; 2]); assert!(!tangent.requires_grad()); }
        else if operation == 4 { assert_eq!(tangent.id(), seed.id()); }
        else {
            assert_eq!(tangent.dtype(), dtype);
            assert!(tangent.requires_grad());
            let actual = tangent.sum().grad_wrt(&[&seed], false).unwrap().remove(0).to_vec_f64();
            for (actual, y) in actual.into_iter().zip(output.to_vec_f64()) {
                assert!((actual + 0.5 * y.powi(3)).abs() <= 2e-7);
            }
        }
    } }
}

#[test]
fn empty_offsets_capacity_and_replay_are_checked_without_wrong_reads() {
    for dtype in [DType::F32, DType::F64] { for operation in 0..6 {
        let base = Tensor::zeros(&[0, 8]).to_dtype(dtype).requires_grad_(true).unwrap();
        let x = base.narrow(1, 4, 4).unwrap();
        let y = transform(&x, operation);
        assert_eq!(y.shape(), &[0, 4]);
        assert!(y.to_vec_f64().is_empty());
        y.sum().backward();
        assert_eq!(base.grad().unwrap().shape(), base.shape());
    } }
    let huge = Tensor::scalar(1.).expand(&[isize::MAX as usize / 4 + 1]).unwrap();
    assert!(huge.floor().is_err() && huge.rsqrt().is_err());
    let input = Tensor::from_vec(vec![0.25, 1.75], &[2]).unwrap();
    let captured = ferro_core::capture(|| input.round_decimals(1)).unwrap();
    let chain = ferro_core::graph::CompiledChain::compile(&captured).unwrap();
    input.copy_from(&Tensor::from_vec(vec![0.35, 1.85], &[2]).unwrap()).unwrap();
    assert_eq!(chain.replay().unwrap().to_vec(), vec![0.4, 1.8]);
}

#[test]
fn graph_rsqrt_uses_saved_output_after_input_changes_without_a_cycle() {
    let input = Tensor::from_vec(vec![0.5, 1.5], &[2]).unwrap().requires_grad_(true).unwrap();
    let output = input.rsqrt().unwrap();
    let initial = output.sum().grad_wrt(&[&input], false).unwrap().remove(0);
    with_grad_enabled(false, || input.copy_leaf_from(&Tensor::full(&[2], 4.))).unwrap();
    let first = output.sum().grad_wrt(&[&input], true).unwrap().remove(0);
    assert_eq!(first.to_vec(), initial.to_vec());
    let expected = output.to_vec().into_iter().map(|y| 0.75 * y.powi(5)).collect::<Vec<_>>();
    let second = first.sum().grad_wrt(&[&input], false).unwrap().remove(0);
    for (actual, expected) in second.to_vec().into_iter().zip(expected) { assert!((actual - expected).abs() < 1e-6); }
    output.detach().copy_from(&Tensor::ones(&[2])).unwrap();
    assert!(first.sum().grad_wrt(&[&input], false).is_err());
    assert!(input.grad().is_none());
}
