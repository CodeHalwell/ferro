use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::DualLevel;

fn leaf(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(values.to_vec(), shape).unwrap().requires_grad_(true).unwrap()
}
fn mask(values: &[bool], shape: &[usize]) -> Tensor { Tensor::from_vec_bool(values.to_vec(), shape).unwrap() }
fn equal(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!(a == b || a.is_nan() && b.is_nan(), "{a} != {b}"); }
}

#[test]
fn condition_is_nondifferentiable_but_both_data_adjoints_are_present() {
    let condition = leaf(&[0., 0.], &[2]);
    let a = leaf(&[2., 3.], &[2]);
    let b = leaf(&[4., 5.], &[2]);
    let y = Tensor::where_cond(&condition, &a, &b).unwrap();
    let gradients = y.vjp_wrt_optional(&[&condition, &a, &b], &Tensor::ones(&[2]), true).unwrap();
    assert!(gradients[0].is_none());
    assert_eq!(gradients[1].as_ref().unwrap().to_vec(), [0., 0.]);
    assert!(!gradients[1].as_ref().unwrap().is_structural_zero());
    assert_eq!(gradients[2].as_ref().unwrap().to_vec(), [1., 1.]);
    y.sum().backward();
    assert!(condition.grad().is_none());
    assert_eq!(a.grad().unwrap().to_vec(), [0., 0.]);
    assert!(!Tensor::where_cond(&condition, &a.detach(), &b.detach()).unwrap().requires_grad());
    let boolean = mask(&[false, false], &[2]);
    Tensor::where_cond(&boolean, &a, &b).unwrap().sum().backward();
    assert!(boolean.grad().is_none());
}

#[test]
fn reverse_saves_only_condition_for_raw_and_graph_gradients() {
    for graph in [false, true] { for changed in 0..4 {
        let condition = mask(&[true, false], &[2]);
        let a = leaf(&[2., 3.], &[2]);
        let b = leaf(&[4., 5.], &[2]);
        let output = Tensor::where_cond(&condition, &a, &b).unwrap();
        [&condition, &a, &b, &output][changed]._bump_version_for_test();
        let result = output.vjp_wrt_optional(&[&a, &b], &Tensor::ones(&[2]), graph);
        if changed == 0 { assert!(result.is_err()); }
        else {
            let result = result.unwrap();
            assert_eq!(result[0].as_ref().unwrap().to_vec(), [1., 0.]);
            assert_eq!(result[1].as_ref().unwrap().to_vec(), [0., 1.]);
        }
    } }
    let condition = mask(&[true, false], &[2]);
    let a = leaf(&[2., 3.], &[2]);
    let b = leaf(&[4., 5.], &[2]);
    let seed = leaf(&[0.2, 0.7], &[2]);
    let output = Tensor::where_cond(&condition, &a, &b).unwrap();
    let derivative = output.vjp_wrt(&[&a], &seed, true).unwrap().remove(0);
    seed._bump_version_for_test();
    assert_eq!(derivative.sum().grad_wrt(&[&seed], false).unwrap()[0].to_vec(), [1., 0.]);
    condition._bump_version_for_test();
    assert!(derivative.sum().grad_wrt(&[&seed], false).is_err());
}

#[test]
fn live_seed_graph_masking_unbroadcast_and_nonfinite_seeds() {
    let condition = mask(&[true, false, true, false, true, false], &[2, 3]);
    let a = leaf(&[0.2, 0.4], &[2, 1]);
    let b = leaf(&[0.5, 0.7, 0.9], &[1, 3]);
    let seed = leaf(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let output = Tensor::where_cond(&condition, &a, &b).unwrap();
    let gradients = output.vjp_wrt(&[&a, &b], &seed, true).unwrap();
    assert_eq!(gradients[0].to_vec(), [4., 5.]);
    assert_eq!(gradients[1].to_vec(), [4., 2., 6.]);
    assert!(!gradients[0].depends_on(&a) && !gradients[0].depends_on(&b));
    let mixed = gradients[0].sum().vjp_wrt_optional(&[&a, &b, &seed], &Tensor::scalar(1.), false).unwrap();
    assert!(mixed[0].is_none() && mixed[1].is_none());
    assert_eq!(mixed[2].as_ref().unwrap().to_vec(), [1., 0., 1., 0., 1., 0.]);
    for value in [f32::NAN, f32::INFINITY] {
        let g = leaf(&[value; 6], &[2, 3]);
        let grad = output.vjp_wrt(&[&a, &b], &g, true).unwrap();
        equal(&grad[0].to_vec_f64(), &[value as f64, value as f64]);
        let aa = leaf(&[1., 2.], &[2]);
        let bb = leaf(&[3., 4.], &[2]);
        let out = Tensor::where_cond(&mask(&[true, false], &[2]), &aa, &bb).unwrap();
        let grad = out.vjp_wrt(&[&aa, &bb], &leaf(&[value; 2], &[2]), true).unwrap();
        equal(&grad[0].to_vec_f64(), &[value as f64, 0.]);
        equal(&grad[1].to_vec_f64(), &[0., value as f64]);
    }
}

#[test]
fn inactive_data_branch_still_receives_a_real_zero_adjoint() {
    let x = leaf(&[0., 2.], &[2]);
    let reciprocal = Tensor::ones(&[2]).div(&x).unwrap();
    let out = Tensor::where_cond(&mask(&[false, true], &[2]), &reciprocal, &x).unwrap();
    let gradient = out.sum().grad_wrt(&[&x], false).unwrap().remove(0);
    equal(&gradient.to_vec_f64(), &[f64::NAN, -0.25]);
}

#[test]
fn direct_jvp_uses_typed_scalar_promotion_and_only_tangent_edges() {
    for left_shape in [vec![], vec![1], vec![3]] { for both in [false, true] { for left_double in [false, true] { for right_double in [false, true] {
        let a = Tensor::ones(&left_shape).requires_grad_(true).unwrap();
        let b = leaf(&[1., 2., 3.], &[3]);
        let va = Tensor::full(&left_shape, 2.).to_dtype(if left_double { DType::F64 } else { DType::F32 }).requires_grad_(true).unwrap();
        let vb = Tensor::full(&[3], 3.).to_dtype(if right_double { DType::F64 } else { DType::F32 }).requires_grad_(true).unwrap();
        let condition = mask(&[true, false, true], &[3]);
        let level = if both { DualLevel::new(&[&a, &b], &[&va, &vb]).unwrap() } else { DualLevel::new(&[&a], &[&va]).unwrap() };
        let output = Tensor::where_cond(&condition, &a, &b).unwrap();
        let dy = level.tangent(&output).unwrap().unwrap();
        drop(level);
        let double = both && right_double || left_double && !left_shape.is_empty();
        assert_eq!(dy.dtype(), if double { DType::F64 } else { DType::F32 });
        equal(&dy.to_vec_f64(), &[2., if both { 3. } else { 0. }, 2.]);
        assert!(dy.requires_grad() && !dy.depends_on(&a) && !dy.depends_on(&b));
        a._bump_version_for_test(); b._bump_version_for_test(); output._bump_version_for_test();
        let grad = dy.sum().grad_wrt(&[&va], false).unwrap().remove(0);
        assert_eq!(grad.dtype(), va.dtype());
        equal(&grad.to_vec_f64(), if left_shape == [3] { &[1., 0., 1.] } else { &[2.] });
    } } } }
}

#[test]
fn jvp_no_grad_flags_and_condition_versions() {
    for enabled in [false, true] { for live in [false, true] {
        let a = leaf(&[1., 2.], &[2]);
        let b = leaf(&[3., 4.], &[2]);
        let v = Tensor::ones(&[2]).requires_grad_(live).unwrap();
        let condition = mask(&[true, false], &[2]);
        let level = DualLevel::new(&[&a], &[&v]).unwrap();
        let output = with_grad_enabled(enabled, || Tensor::where_cond(&condition, &a, &b).unwrap());
        let dy = level.tangent(&output).unwrap().unwrap();
        drop(level);
        assert_eq!(output.requires_grad(), enabled);
        assert_eq!(dy.requires_grad(), enabled && live);
        assert_eq!(dy.to_vec(), [1., 0.]);
        if dy.requires_grad() {
            condition._bump_version_for_test();
            assert!(dy.sum().grad_wrt(&[&v], false).is_err());
        }
    } }
    let condition = leaf(&[1., 0.], &[2]);
    let level = DualLevel::new(&[&condition], &[&Tensor::ones(&[2])]).unwrap();
    let output = Tensor::where_cond(&condition, &Tensor::ones(&[2]), &Tensor::zeros(&[2])).unwrap();
    assert!(level.tangent(&output).unwrap().is_none());
    assert!(!output.requires_grad());
}

#[test]
fn empty_broadcast_vjp_and_typed_jvp_skip_irrelevant_huge_operands() {
    let size = 1_000_000_000_000usize;
    for dtype in [DType::F32, DType::F64] {
        let condition = mask(&[], &[1, 0]).expand(&[size, 0]).unwrap();
        let a = Tensor::scalar(2.).to_dtype(dtype).expand(&[size, 1]).unwrap().requires_grad_(true).unwrap();
        let b = Tensor::zeros(&[1, 0]).to_dtype(dtype).requires_grad_(true).unwrap();
        let out = Tensor::where_cond(&condition, &a, &b).unwrap();
        let seed = Tensor::zeros(&[size, 0]).to_dtype(dtype).requires_grad_(true).unwrap();
        let gradients = out.vjp_wrt_preserve(&[&a, &b], &seed).unwrap();
        assert_eq!(gradients[0].shape(), &[size, 1]);
        assert_eq!(gradients[0].strides(), &[0, 1]);
        assert_eq!(gradients[1].shape(), &[1, 0]);
        if dtype == DType::F32 {
            let vector = Tensor::scalar(3.).to_dtype(DType::F64).expand(&[size, 1]).unwrap().requires_grad_(true).unwrap();
            let level = DualLevel::new(&[&a], &[&vector]).unwrap();
            let out = Tensor::where_cond(&condition, &a, &b).unwrap();
            let dy = level.tangent(&out).unwrap().unwrap();
            assert_eq!(dy.dtype(), DType::F64); assert_eq!(dy.shape(), &[size, 0]);
        }
    }
}

#[test]
fn values_keep_typed_storage_strides_and_existing_scope_with_checked_capacity() {
    let cond = mask(&[true, false, false, true], &[2, 2]).transpose(0, 1).unwrap();
    let a = Tensor::from_vec_i64(vec![i64::MIN, 2, 3, i64::MAX], &[2, 2]).unwrap().transpose(0, 1).unwrap();
    let b = Tensor::from_vec_i64(vec![7], &[]).unwrap();
    assert_eq!(Tensor::where_cond(&cond, &a, &b).unwrap().to_vec_i64(), [i64::MIN, 7, 7, i64::MAX]);
    let huge = Tensor::ones(&[1]).expand(&[isize::MAX as usize / 4 + 1]).unwrap();
    assert!(Tensor::where_cond(&mask(&[true], &[]), &huge, &huge).is_err());
    let promoted = Tensor::where_cond(&cond, &Tensor::ones(&[2, 2]), &a).unwrap();
    assert_eq!(promoted.dtype(), DType::F32);
    assert_eq!(promoted.to_vec(), vec![1., 3., 2., 1.]);
    let captured = ferro_core::capture(|| Tensor::where_cond(&cond, &Tensor::ones(&[2, 2]), &Tensor::zeros(&[2, 2])));
    assert!(captured.is_err());
}
