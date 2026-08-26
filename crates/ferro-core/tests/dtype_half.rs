//! F16/BF16 tensor semantics: bit-preserving construction and access, casts
//! through to_dtype (RNE), materialization, view gathering, and the
//! f32-only-compute policy (kernels and autograd refuse half tensors until
//! an explicit cast). Conversion math itself is unit-tested in src/half.rs.

use ferro_core::{DType, Tensor};

#[test]
fn f16_bits_round_trip_and_materialize() {
    // 1.0, -2.5, 65504 (max), smallest subnormal, +inf, quiet NaN.
    let bits = vec![0x3C00u16, 0xC100, 0x7BFF, 0x0001, 0x7C00, 0x7E00];
    let t = Tensor::from_vec_f16_bits(bits.clone(), &[6]).unwrap();
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.to_vec_f16_bits().unwrap(), bits);
    let v = t.to_vec();
    assert_eq!(&v[..3], &[1.0, -2.5, 65504.0]);
    assert_eq!(v[3], 5.9604645e-8);
    assert_eq!(v[4], f32::INFINITY);
    assert!(v[5].is_nan());
    // f64 materialization is exact through f32.
    assert_eq!(t.to_vec_f64()[2], 65504.0);
    assert_eq!(t.to_vec_i64()[1], -2);
}

#[test]
fn bf16_bits_round_trip_and_materialize() {
    let bits = vec![0x3F80u16, 0xC020, 0x7F7F]; // 1.0, -2.5, bf16 max
    let t = Tensor::from_vec_bf16_bits(bits.clone(), &[3]).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.to_vec_bf16_bits().unwrap(), bits);
    assert_eq!(t.to_vec(), vec![1.0, -2.5, 3.3895314e38]);
}

#[test]
fn to_dtype_casts_round_to_nearest_even_and_back_exactly() {
    // f32 -> f16: 1/3 rounds to the nearest f16; the cast back is exact.
    let x = Tensor::from_vec(vec![1.0 / 3.0, 1.0, -2.5], &[3]).unwrap();
    let h = x.to_dtype(DType::F16);
    assert_eq!(h.dtype(), DType::F16);
    assert_eq!(h.to_vec_f16_bits().unwrap(), vec![0x3555, 0x3C00, 0xC100]);
    assert_eq!(h.to_dtype(DType::F32).to_vec(), vec![0.33325195, 1.0, -2.5]);

    // Same-dtype cast is bit-preserving (every half value is exact in f32).
    let h2 = h.to_dtype(DType::F16);
    assert_eq!(h2.to_vec_f16_bits().unwrap(), h.to_vec_f16_bits().unwrap());

    // f32 -> bf16 keeps the top half.
    let b = x.to_dtype(DType::BF16);
    assert_eq!(b.to_vec_bf16_bits().unwrap()[1], 0x3F80);
    // Cross-half cast goes through f32: f16(1/3) -> bf16 re-rounds.
    let hb = h.to_dtype(DType::BF16);
    assert_eq!(hb.dtype(), DType::BF16);
    assert_eq!(hb.to_vec()[0], f32::from_bits(0x3EAB_0000)); // bf16(0.33325195)
}

#[test]
fn views_gather_half_storage() {
    let t = Tensor::from_vec_f16_bits(vec![0x3C00, 0x4000, 0x4200, 0x4400], &[2, 2]).unwrap();
    // transpose on a no-grad tensor is a pure view; bit access gathers it.
    let tr = t.transpose(0, 1).unwrap();
    assert_eq!(
        tr.to_vec_f16_bits().unwrap(),
        vec![0x3C00, 0x4200, 0x4000, 0x4400]
    );
    assert_eq!(tr.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn half_tensors_are_data_not_compute() {
    let h = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap().to_dtype(DType::F16);
    // Kernels are f32-only: explicit cast required.
    assert!(h.add(&h).is_err());
    assert!(h.requires_grad_(true).is_err());
    assert!(h.zero_().is_err(), "in-place is f32-only too");
    // The documented route into math.
    let f = h.to_dtype(DType::F32);
    assert_eq!(f.add(&f).unwrap().to_vec(), vec![2.0, 4.0]);
}

#[test]
fn bits_accessors_refuse_other_dtypes() {
    let x = Tensor::from_vec(vec![1.0], &[1]).unwrap();
    assert!(x.to_vec_f16_bits().is_err());
    assert!(x.to_vec_bf16_bits().is_err());
    let h = x.to_dtype(DType::F16);
    assert!(h.to_vec_bf16_bits().is_err());
}
