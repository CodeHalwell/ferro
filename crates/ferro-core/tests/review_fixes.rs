use ferro_core::testkit::{grad_check, GradTol};
use ferro_core::Tensor;

#[test]
fn sum_backward_scales_by_upstream_grad() {
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    x.sum().mul(&Tensor::scalar(2.0)).unwrap().backward();
    let g = x.grad().unwrap().to_vec();
    assert!(g.iter().all(|&v| (v - 2.0).abs() < 1e-6), "dx {g:?}");
}

#[test]
fn mean_backward_scales_by_upstream_grad_over_n() {
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    // loss = 3 * mean(x); dx = 3/4 everywhere
    x.mean().mul(&Tensor::scalar(3.0)).unwrap().backward();
    let g = x.grad().unwrap().to_vec();
    assert!(
        g.iter().all(|&v| (v - 0.75).abs() < 1e-6),
        "dx {g:?} want 0.75"
    );
}

#[test]
fn scaled_sum_grad_check() {
    let x = Tensor::from_vec((0..8).map(|i| ((i as f32) * 0.7).sin()).collect(), &[2, 4])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    grad_check(&[x], |t: &[Tensor]| {
        t[0].sum().mul(&Tensor::scalar(1.3)).unwrap()
    });
    let _ = GradTol::Default;
}

#[test]
fn bf16_round_tiny_negative_is_negative_zero_not_inf() {
    let v = ferro_core::amp::bf16_round(-1e-41);
    assert_eq!(v.to_bits(), 0x8000_0000, "-1e-41 must round to -0.0");
}

#[test]
fn bf16_round_saturates_with_sign_on_overflow() {
    let big = f32::from_bits(0x7F7F_FFFF);
    assert_eq!(
        ferro_core::amp::bf16_round(big).to_bits(),
        f32::INFINITY.to_bits()
    );
    assert_eq!(
        ferro_core::amp::bf16_round(-big).to_bits(),
        f32::NEG_INFINITY.to_bits()
    );
}

#[test]
fn bf16_round_is_round_to_nearest_even_on_exact_ties() {
    // dropped mantissa == 0x8000 exactly: tie. RNE rounds to even kept LSB.
    let tie_even = f32::from_bits(0x3FFE_8000);
    assert_eq!(
        ferro_core::amp::bf16_round(tie_even).to_bits(),
        0x3FFE_0000,
        "tie with even kept LSB rounds down"
    );
    let tie_odd = f32::from_bits(0x3FFF_8000);
    assert_eq!(
        ferro_core::amp::bf16_round(tie_odd).to_bits(),
        0x4000_0000,
        "tie with odd kept LSB rounds up"
    );
}
