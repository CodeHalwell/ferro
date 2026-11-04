use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::DualLevel;
use ferro_core::ops_ext::add_alpha::Alpha;

fn leaf(values: &[f32], shape: &[usize]) -> Tensor { Tensor::from_vec(values.to_vec(), shape).unwrap().requires_grad_(true).unwrap() }
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a-b).abs() < 2e-5, "{a} != {b}"); }
}

#[test]
fn fused_float_primal_matches_torch_cancellation_and_overflow_goldens() {
    for length in [1, 2, 8, 16, 33] {
        let a = Tensor::from_vec(vec![-1e30; length], &[length]).unwrap();
        let b = Tensor::from_vec(vec![1e30; length], &[length]).unwrap();
        let alpha = Alpha::F64(1.0000001192092896);
        assert_eq!(a.add_alpha(&b, alpha).unwrap().to_vec(), vec![1.19209291344579e23f32; length]);
        let a64 = Tensor::from_vec_f64(vec![-1e30; length], &[length]).unwrap();
        let b64 = Tensor::from_vec_f64(vec![1e30; length], &[length]).unwrap();
        assert_eq!(a64.add_alpha(&b64, alpha).unwrap().to_vec_f64(), vec![1.1920928955078125e23; length]);
    }
    let a = Tensor::from_vec(vec![f32::NEG_INFINITY, f32::INFINITY], &[2]).unwrap();
    let b = Tensor::from_vec(vec![1e30, -1e30], &[2]).unwrap();
    assert_eq!(a.add_alpha(&b, Alpha::F64(1e10)).unwrap().to_vec(), vec![f32::NEG_INFINITY, f32::INFINITY]);
    let negative = b.neg();
    assert_eq!(a.sub_alpha(&negative, Alpha::F64(1e10)).unwrap().to_vec(), a.to_vec());
}

#[test]
fn integral_categories_wrapping_and_scalar_validation_are_explicit() {
    let a = Tensor::from_vec_i64(vec![i64::MAX, 1], &[2]).unwrap();
    let b = Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap();
    assert_eq!(a.add_alpha(&b, Alpha::I64(2)).unwrap().to_vec_i64(), vec![i64::MIN+1, 5]);
    assert_eq!(a.sub_alpha(&b, Alpha::U64(u64::MAX)).unwrap().to_vec_i64(), vec![i64::MIN, 3]);
    assert!(a.add_alpha(&b, Alpha::U64(u64::MAX)).is_err());
    for alpha in [Alpha::Bool(true), Alpha::F64(2.)] { assert!(a.add_alpha(&b, alpha).is_err()); }
    let p = Tensor::from_vec_bool(vec![true, false], &[2]).unwrap();
    let q = Tensor::from_vec_bool(vec![false, true], &[2]).unwrap();
    assert_eq!(p.add_alpha(&q, Alpha::Bool(false)).unwrap().to_vec_bool(), vec![true, false]);
    assert_eq!(p.add_alpha(&q, Alpha::U64(u64::MAX)).unwrap().to_vec_bool(), vec![true, true]);
    assert!(p.add_alpha(&q, Alpha::F64(1.)).is_err());
    assert!(p.sub_alpha(&q, Alpha::I64(1)).is_err());
    assert!(p.sub_alpha(&Tensor::ones(&[2]), Alpha::I64(1)).is_err());
    let f = Tensor::ones(&[2]);
    assert!(f.add_alpha(&f, Alpha::Bool(true)).is_err());
    for alpha in [1e100, 3.4028235e38] { assert!(f.add_alpha(&f, Alpha::F64(alpha)).is_err()); }
    assert!(f.add_alpha(&f, Alpha::F64(f64::INFINITY)).unwrap().to_vec().iter().all(|x| x.is_infinite()));
    assert!(f.add_alpha(&f, Alpha::F64(f64::NAN)).unwrap().to_vec().iter().all(|x| x.is_nan()));
}

#[test]
fn promotion_precedes_alpha_conversion_and_preserves_mixed_gradients() {
    let a = leaf(&[1., 2.], &[2]);
    let b = Tensor::from_vec_f64(vec![0.5, -0.3], &[2]).unwrap().requires_grad_(true).unwrap();
    let out = a.add_alpha(&b, Alpha::F64(1e100)).unwrap();
    assert_eq!(out.dtype(), DType::F64);
    out.sum().backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![1., 1.]);
    assert_eq!(b.grad().unwrap().to_vec_f64(), vec![1e100; 2]);
    let scalar = Tensor::from_vec_f64(vec![1.], &[]).unwrap();
    assert!(a.add_alpha(&scalar, Alpha::F64(1e100)).is_err());
    assert_eq!(a.add_alpha(&scalar, Alpha::F64(0.5)).unwrap().dtype(), DType::F32);
    let i = Tensor::from_vec_i64(vec![2, 3], &[2]).unwrap();
    assert_eq!(i.add_alpha(&a, Alpha::F64(0.5)).unwrap().to_vec(), vec![2.5, 4.]);
}

#[test]
fn broadcast_strides_ordinary_gradients_and_finite_differences() {
    let a = leaf(&[0.2, -0.3, 0.6, 0.1, -0.2, 0.4], &[3, 2]);
    let b = leaf(&[0.7, -0.4, 0.2], &[3]);
    let seed = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
    let out = a.transpose(0, 1).unwrap().sub_alpha(&b, Alpha::F64(1.25)).unwrap();
    let gradients = out.vjp_wrt(&[&a, &b], &seed, false).unwrap();
    assert_eq!(gradients[0].to_vec(), vec![1., 4., 2., 5., 3., 6.]);
    assert_eq!(gradients[1].to_vec(), vec![-6.25, -8.75, -11.25]);
    ferro_core::testkit::grad_check_strict(&[a.detach(), b.detach()], |v| v[0].transpose(0, 1).unwrap().add_alpha(&v[1], Alpha::F64(-0.75)).unwrap().powf(2.).sum());
}

#[test]
fn linear_adjoint_reads_no_saved_values_and_retains_live_seed_provenance() {
    let a = leaf(&[0.2, -0.3], &[2]);
    let b = leaf(&[0.7, -0.4], &[2]);
    let seed = leaf(&[0.4, -0.8], &[2]);
    let out = a.add_alpha(&b, Alpha::F64(1.5)).unwrap();
    a.detach().copy_from(&Tensor::ones(&[2])).unwrap();
    b.detach().copy_from(&Tensor::ones(&[2])).unwrap();
    out.detach().copy_from(&Tensor::zeros(&[2])).unwrap();
    let gradients = out.vjp_wrt(&[&a, &b], &seed, true).unwrap();
    assert!(gradients[0].depends_on(&seed) && gradients[1].depends_on(&seed));
    assert!(!gradients[0].depends_on(&a) && !gradients[1].depends_on(&b));
    close(&gradients[1].to_vec(), &[0.6, -1.2]);
    assert_eq!(gradients[1].sum().grad_wrt(&[&seed], false).unwrap()[0].to_vec(), vec![1.5; 2]);
    out.vjp_wrt(&[&a, &b], &seed, false).unwrap();
    assert!(a.grad().is_none() && b.grad().is_none());
    let cube = a.add_alpha(&b, Alpha::F64(2.)).unwrap().powf(3.).sum();
    let first = cube.grad_wrt(&[&a], true).unwrap().remove(0);
    let mixed = first.sum().grad_wrt(&[&b], true).unwrap().remove(0);
    assert_eq!(mixed.to_vec(), vec![36.; 2]);
    assert_eq!(mixed.sum().grad_wrt(&[&a], false).unwrap()[0].to_vec(), vec![12.; 2]);
}

#[test]
fn direct_forward_rule_preserves_original_alpha_and_independent_tangent_edges() {
    let a = leaf(&[0.2, -0.3], &[2]);
    let b = leaf(&[0.7, -0.4], &[2]);
    let va = Tensor::from_vec_f64(vec![1., 2.], &[2]).unwrap().requires_grad_(true).unwrap();
    let vb = Tensor::from_vec_f64(vec![3., 4.], &[2]).unwrap().requires_grad_(true).unwrap();
    let alpha = Alpha::F64(1.0000000000000002);
    let level = DualLevel::new(&[&a, &b], &[&va, &vb]).unwrap();
    let out = a.add_alpha(&b, alpha).unwrap();
    let tangent = level.tangent(&out).unwrap().unwrap();
    drop(level);
    assert_eq!(tangent.to_vec_f64(), vec![4.000000000000001, 6.000000000000001]);
    assert!(tangent.depends_on(&va) && tangent.depends_on(&vb));
    assert!(!tangent.depends_on(&a) && !tangent.depends_on(&b));
    tangent.sum().backward();
    assert_eq!(va.grad().unwrap().to_vec_f64(), vec![1.; 2]);
    assert_eq!(vb.grad().unwrap().to_vec_f64(), vec![1.0000000000000002; 2]);
    let level = DualLevel::new(&[&a], &[&va]).unwrap();
    let out = a.add_alpha(&b, Alpha::F64(f64::INFINITY)).unwrap();
    assert_eq!(level.tangent(&out).unwrap().unwrap().to_vec_f64(), va.to_vec_f64());
    drop(level);
    let level = DualLevel::new(&[&b], &[&vb]).unwrap();
    let out = a.sub_alpha(&b, alpha).unwrap();
    assert_eq!(level.tangent(&out).unwrap().unwrap().to_vec_f64(), vec![-3.000000000000001, -4.000000000000001]);
}

#[test]
fn torch_signed_minimum_contract_and_unsigned_ad_rejection_are_preserved() {
    let a = leaf(&[1.], &[1]);
    let b = leaf(&[1.], &[1]);
    let out = a.sub_alpha(&b, Alpha::I64(i64::MIN)).unwrap();
    assert_eq!(out.to_vec(), vec![i64::MIN as f32]);
    out.backward();
    assert_eq!(b.grad().unwrap().to_vec(), vec![-(i64::MIN as f32)]);
    let v = Tensor::from_vec_f64(vec![3.], &[1]).unwrap();
    let level = DualLevel::new(&[&b], &[&v]).unwrap();
    let out = a.sub_alpha(&b, Alpha::I64(i64::MIN)).unwrap();
    assert_eq!(level.tangent(&out).unwrap().unwrap().to_vec_f64(), vec![2.7670116110564327e19]);
    let invalid_tangent = a.add_alpha(&b, Alpha::U64(u64::MAX)).unwrap();
    assert!(level.tangent(&invalid_tangent).is_err());
    drop(level);
    let output = a.sub_alpha(&b, Alpha::U64(u64::MAX)).unwrap();
    assert_eq!(output.to_vec(), vec![2.]);
    for graph in [false, true] {
        assert_eq!(output.sum().grad_wrt(&[&a], graph).unwrap()[0].to_vec(), vec![1.]);
        assert!(output.sum().grad_wrt(&[&b], graph).is_err());
        assert!(output.sum().grad_wrt(&[&a, &b], graph).is_err());
    }
    let before = (a.grad().unwrap().to_vec(), b.grad().unwrap().to_vec());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| output.sum().backward())).is_err());
    assert_eq!((a.grad().unwrap().to_vec(), b.grad().unwrap().to_vec()), before);
    let inactive = b.detach();
    a.zero_grad();
    a.add_alpha(&inactive, Alpha::U64(u64::MAX)).unwrap().sum().backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![1.]);
    assert_eq!(with_grad_enabled(false, || a.sub_alpha(&b, Alpha::U64(u64::MAX))).unwrap().to_vec(), vec![2.]);
}

#[test]
fn empty_outputs_validate_before_allocation_and_backward_keeps_zero_shapes() {
    let a = leaf(&[], &[0, 3]);
    let b = leaf(&[1., 2., 3.], &[3]);
    let out = a.add_alpha(&b, Alpha::F64(1.5)).unwrap();
    assert_eq!(out.shape(), &[0, 3]);
    out.sum().backward();
    assert_eq!(a.grad().unwrap().shape(), &[0, 3]);
    assert_eq!(b.grad().unwrap().to_vec(), vec![0.; 3]);
    assert!(a.add_alpha(&b, Alpha::F64(1e100)).is_err());
    assert!(ferro_core::capture(|| a.add_alpha(&b, Alpha::I64(1))).is_err());
    assert!(a.add_alpha(&Tensor::zeros(&[4]), Alpha::I64(1)).is_err());
    let huge = Tensor::scalar(1.).expand(&[1 << 30, 1 << 30]).unwrap();
    assert!(huge.add_alpha(&huge, Alpha::I64(1)).is_err());
}

#[test]
fn huge_empty_broadcast_finishes_without_reading_or_casting_expanded_input() {
    const FLAG: &str = "FERRO_ALPHA_EMPTY_CHILD";
    if std::env::var_os(FLAG).is_some() {
        let base = leaf(&[1.], &[]);
        let huge = base.expand(&[1usize << 60, 1]).unwrap();
        let empty = Tensor::from_vec_f64(Vec::new(), &[1, 0]).unwrap().requires_grad_(true).unwrap();
        let output = huge.add_alpha(&empty, Alpha::F64(1e100)).unwrap();
        assert_eq!(output.shape(), &[1usize << 60, 0]);
        assert_eq!(output.dtype(), DType::F64);
        assert!(output.to_vec_f64().is_empty());
        assert!(output.depends_on(&base) && output.depends_on(&empty));
        return;
    }
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "huge_empty_broadcast_finishes_without_reading_or_casting_expanded_input"])
        .env(FLAG, "1").stdout(std::process::Stdio::null()).spawn().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() { assert!(status.success()); break; }
        if std::time::Instant::now() > deadline { child.kill().unwrap(); child.wait().unwrap(); panic!("empty alpha arithmetic exceeded bounded runtime"); }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let a = leaf(&[], &[0, 2]);
    let b = Tensor::from_vec_f64(vec![2., 3.], &[2]).unwrap().requires_grad_(true).unwrap();
    let output = a.add_alpha(&b, Alpha::F64(1.25)).unwrap();
    output.sum().backward();
    assert_eq!(a.grad().unwrap().dtype(), DType::F32);
    assert_eq!(a.grad().unwrap().shape(), &[0, 2]);
    assert_eq!(b.grad().unwrap().to_vec_f64(), vec![0.; 2]);
}

#[test]
fn torch_jvp_is_separately_rounded_and_subtraction_adjoint_keeps_signed_zero() {
    let a = leaf(&[0.2], &[1]);
    let b = leaf(&[0.7], &[1]);
    let va = leaf(&[-1e30], &[1]);
    let vb = leaf(&[1e30], &[1]);
    let level = DualLevel::new(&[&a, &b], &[&va, &vb]).unwrap();
    let output = a.add_alpha(&b, Alpha::F64(1.0000001192092896)).unwrap();
    assert_eq!(level.tangent(&output).unwrap().unwrap().to_vec(), vec![1.5111572745182865e23f32]);
    drop(level);
    let seed = leaf(&[2.], &[1]);
    let output = a.add_alpha(&b, Alpha::I64(1)).unwrap();
    assert_eq!(output.vjp_wrt(&[&b], &seed, true).unwrap()[0].id(), seed.id());
    let output = a.sub_alpha(&b, Alpha::I64(0)).unwrap();
    assert_eq!(output.vjp_wrt(&[&b], &seed, true).unwrap()[0].to_vec()[0].to_bits(), (-0.0f32).to_bits());
    let double = Tensor::from_vec_f64(vec![1.], &[1]).unwrap().requires_grad_(true).unwrap();
    let output = a.add_alpha(&double, Alpha::F64(1.5)).unwrap();
    a.detach().copy_from(&Tensor::zeros(&[1])).unwrap();
    output.sum().backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![1.]);
    assert_eq!(double.grad().unwrap().to_vec_f64(), vec![1.5]);
    let cast = a.to_dtype_autograd(DType::F64, false).unwrap();
    let nonlinear = cast.mul(&cast).unwrap();
    cast.detach().copy_from(&Tensor::from_vec_f64(vec![9.], &[1]).unwrap()).unwrap();
    assert!(nonlinear.grad_wrt(&[&a], false).is_err());
}

#[test]
fn forward_rule_keeps_scaling_before_broadcast_for_reverse_over_forward() {
    let a = leaf(&[0.1, 0.2, 0.3], &[3]);
    let b = leaf(&[0.4], &[1]);
    let va = leaf(&[0., 0., 0.], &[3]);
    let vb = leaf(&[0.5], &[1]);
    let level = DualLevel::new(&[&a, &b], &[&va, &vb]).unwrap();
    let out = a.add_alpha(&b, Alpha::F64(1e10)).unwrap();
    let tangent = level.tangent(&out).unwrap().unwrap();
    drop(level);
    let seed = Tensor::from_vec(vec![1e30, -1e30, 3.], &[3]).unwrap();
    assert_eq!(tangent.vjp_wrt(&[&vb], &seed, false).unwrap()[0].to_vec(), vec![3e10]);
}
