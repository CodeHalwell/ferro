use super::*;
use std::cell::Cell;
use ferro_core::dispatch::Backend;

thread_local! { pub(super) static PACKED_FAULT: Cell<u8> = const { Cell::new(0) }; }
#[test]
fn packed_nan_infinity_and_finite_corruption_cannot_escape_bmm() {
    let _reset = Reset;
    for fault in [2, 3, 4] {
        PACKED_FAULT.set(fault);
        let bad = matmul_batch_packed(&[2.0], &[3.0], 1, 1, 1, 1);
        assert!(bad[0].is_nan() || bad[0] != 6.0, "injection inactive");
        assert_eq!(matmul_batch(&[2.0], &[3.0], 1, 1, 1, 1), vec![6.0]);
        assert_eq!(elementwise::FastCpuBackend.matmul_batch(&[2.0], &[3.0], 1, 1, 1, 1), vec![6.0]);
    }
}

#[test]
fn conservative_specials_and_shape_validation() {
    for a in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, f32::from_bits(1), f32::MAX] {
        for b in [0.0, 1.0, -1.0, f32::INFINITY, f32::from_bits(1)] {
            let want = 0.0f32 + a*b;
            let got = matmul_batch(&[a], &[b], 1, 1, 1, 1)[0];
            assert!(got.is_nan() && want.is_nan() || got.to_bits() == want.to_bits());
        }
    }
    for shape in [(0, 3, 4, 5), (2, 0, 4, 5), (2, 3, 4, 0)] {
        assert!(matmul_batch(&[], &[], shape.0, shape.1, shape.2, shape.3).is_empty());
    }
    assert_eq!(matmul_batch(&[], &[], 2, 3, 0, 5), vec![0.0; 30]);
    for shape in [(1, 1, 1, 1), (usize::MAX, 2, 1, 1), (1, 1, usize::MAX, 2), (1, 2, 0, usize::MAX)] {
        assert!(std::panic::catch_unwind(|| matmul_batch(&[], &[], shape.0, shape.1, shape.2, shape.3)).is_err());
    }
}

#[test]
fn install_only_default_backend_bmm_cannot_reenter_packed_kernel() {
    let _reset = Reset;
    install();
    PACKED_FAULT.set(1);
    assert_eq!(ferro_core::CpuBackend.matmul_batch(&vec![1.0;128], &vec![1.0;128], 1, 1, 128, 1), vec![128.0]);
}

#[test]
fn historical_dot_omission_is_contained_bit_for_bit() {
    let _reset = Reset;
    let input = |seed: u64| {
        let mut state = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        (0..16*128*128).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 4.0
        }).collect::<Vec<_>>()
    };
    let aa = input(2556);
    let bb = input(3556);
    let a = &aa[(14*128+14)*128..(14*128+15)*128];
    let b: Vec<_> = (0..128).map(|p| bb[(14*128+p)*128+76]).collect();
    PACKED_FAULT.set(1);
    assert_eq!(matmul_batch_packed(a, &b, 1, 1, 128, 1)[0].to_bits(), 1094337833);
    assert_eq!(matmul_batch(a, &b, 1, 1, 128, 1)[0].to_bits(), 1094681645);
}

struct Reset;
impl Drop for Reset { fn drop(&mut self) { PACKED_FAULT.set(0); } }

#[test]
fn omitted_product_in_packed_kernel_cannot_escape_bmm() {
    let _reset = Reset;
    let a = vec![1.0; 128];
    let b = vec![1.0; 128];
    PACKED_FAULT.set(1);
    let mut suspect = vec![0.0];
    suspect.copy_from_slice(&matmul_batch_packed(&a, &b, 1, 1, 128, 1));
    assert_eq!(suspect, vec![127.0], "injection must reach real packed arithmetic");
    assert_eq!(matmul_batch(&a, &b, 1, 1, 128, 1), vec![128.0]);
    assert_eq!(elementwise::FastCpuBackend.matmul_batch(&a, &b, 1, 1, 128, 1), vec![128.0]);
}
