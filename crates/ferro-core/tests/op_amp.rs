use ferro_core::amp::{amp_matmul, Autocast, OpClass};
use ferro_core::testkit::grad_check_strict;
use ferro_core::Tensor;

#[test]
fn bf16_quantization_matches_reference_values() {
    let t = Tensor::from_vec(vec![1.234_567_9, -0.5, 3.141_592_7, 0.0], &[4]).unwrap();
    let got = t.cast_to_bf16().unwrap().to_vec();
    // bf16 keeps 8 mantissa bits total: 1.234375, -0.5, 3.140625, 0.0
    let want = [1.234_375, -0.5, 3.140_625, 0.0];
    assert_eq!(got, want);
}

#[test]
fn cast_back_returns_f32_master_copy() {
    let t = Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
    let back = t.cast_to_bf16().unwrap().cast_back().unwrap();
    assert_eq!(back.dtype(), ferro_core::DType::F32);
    assert_ne!(back.to_vec(), vec![0.0, 0.0]);
}

#[test]
fn autocast_disabled_passes_tensors_through() {
    let t = Tensor::from_vec(vec![1.234_567_9], &[1]).unwrap();
    let ctx = Autocast { enabled: false };
    let outs = ctx.enter(OpClass::Matmul, &[&t]).unwrap();
    assert_eq!(outs[0].to_vec(), vec![1.234_567_9]);
    let fp32 = ctx.enter(OpClass::Fp32, &[&t]).unwrap();
    assert_eq!(fp32[0].to_vec(), vec![1.234_567_9]);
}

#[test]
fn autocast_enabled_casts_matmul_inputs_but_not_fp32_class() {
    let t = Tensor::from_vec(vec![1.234_567_9], &[1]).unwrap();
    let ctx = Autocast::new();
    let outs = ctx.enter(OpClass::Matmul, &[&t]).unwrap();
    assert_eq!(outs[0].to_vec(), vec![1.234_375]);
    let fp32 = ctx.enter(OpClass::Fp32, &[&t]).unwrap();
    assert_eq!(fp32[0].to_vec(), vec![1.234_567_9]);
}

#[test]
fn amp_matmul_value_matches_f32_matmul_on_exact_inputs() {
    // Inputs exactly representable in bf16: quantization is a no-op, so the
    // fused result must equal plain matmul bit-for-bit up to summation order.
    let a = Tensor::from_vec(vec![0.5, -1.5, 2.0, 1.0, 0.25, -0.75], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 0.5, -2.0, 1.5, 0.25, 0.75], &[3, 2]).unwrap();
    let got = amp_matmul(&a, &b).unwrap();
    let want = a.matmul(&b).unwrap().to_vec();
    for (g, w) in got.to_vec().iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "got {g}, want {w}");
    }
}

#[test]
fn amp_matmul_value_shows_quantization_effect() {
    let a = Tensor::from_vec(vec![1.234_567_9, 2.718_281_7], &[1, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.234_567_9, 3.141_592_7], &[2, 1]).unwrap();
    let got = amp_matmul(&a, &b).unwrap().to_vec()[0];
    let qa = a.cast_to_bf16().unwrap().to_vec();
    let qb = b.cast_to_bf16().unwrap().to_vec();
    let manual = qa[0] * qb[0] + qa[1] * qb[1];
    assert!((got - manual).abs() < 1e-6);
    assert!(got != a.matmul(&b).unwrap().to_vec()[0]);
}

#[test]
fn amp_matmul_grad() {
    // The smooth, grad_checkable core of the autocast path: exact matmul
    // chain rule over pre-quantized operands (straight-through estimator).
    let qa = Tensor::from_vec(vec![0.5, -1.5, 2.0, 1.0], &[2, 2]).unwrap();
    let qb = Tensor::from_vec(vec![1.0, 0.5, -2.0, 1.5], &[2, 2]).unwrap();
    grad_check_strict(&[qa, qb], |t| {
        ferro_core::amp::quantized_matmul(&t[0], &t[1])
            .unwrap()
            .sum()
    });
}

#[test]
fn amp_matmul_master_grads_match_straight_through_reference() {
    use ferro_core::amp::quantized_matmul;
    let a = Tensor::from_vec(vec![0.5, 2.5, -1.5, 1.25], &[2, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let b = Tensor::from_vec(vec![1.25, 0.75, -2.5, 1.5], &[2, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let out = amp_matmul(&a, &b).unwrap();
    out.sum().backward();
    let ga_master = a.grad().unwrap().to_vec();
    let gb_master = b.grad().unwrap().to_vec();

    let qa = a.cast_to_bf16().unwrap().requires_grad_(true).unwrap();
    let qb = b.cast_to_bf16().unwrap().requires_grad_(true).unwrap();
    quantized_matmul(&qa, &qb).unwrap().sum().backward();
    for (gm, gq) in ga_master.iter().zip(qa.grad().unwrap().to_vec()) {
        assert!((gm - gq).abs() < 1e-6);
    }
    for (gm, gq) in gb_master.iter().zip(qb.grad().unwrap().to_vec()) {
        assert!((gm - gq).abs() < 1e-6);
    }
}
