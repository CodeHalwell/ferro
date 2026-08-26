//! Public in-place ops: values, torch-style view aliasing, version-counter
//! bumps, and the safety gates (no autograd history, whole-contiguous
//! destinations, shape/dtype/device agreement). The interaction with
//! backward's version check is in version_counters.rs; optimizer and device
//! behavior are in optim_inplace.rs / optim_device.rs / inplace_device.rs.

use ferro_core::{DType, Tensor};

fn t(v: &[f32]) -> Tensor {
    Tensor::from_vec(v.to_vec(), &[v.len()]).unwrap()
}

#[test]
fn values_for_every_op() {
    let x = t(&[1.0, -2.0, 3.0]);
    x.add_(&t(&[10.0, 20.0, 30.0])).unwrap();
    assert_eq!(x.to_vec(), vec![11.0, 18.0, 33.0]);
    x.sub_(&t(&[1.0, -2.0, 3.0])).unwrap();
    assert_eq!(x.to_vec(), vec![10.0, 20.0, 30.0]);
    x.mul_(&t(&[2.0, 0.5, -1.0])).unwrap();
    assert_eq!(x.to_vec(), vec![20.0, 10.0, -30.0]);
    x.div_(&t(&[2.0, 2.0, 3.0])).unwrap();
    assert_eq!(x.to_vec(), vec![10.0, 5.0, -10.0]);
    x.add_scalar_(1.5).unwrap();
    assert_eq!(x.to_vec(), vec![11.5, 6.5, -8.5]);
    x.mul_scalar_(2.0).unwrap();
    assert_eq!(x.to_vec(), vec![23.0, 13.0, -17.0]);
    x.fill_(7.0).unwrap();
    assert_eq!(x.to_vec(), vec![7.0, 7.0, 7.0]);
    x.zero_().unwrap();
    assert_eq!(x.to_vec(), vec![0.0, 0.0, 0.0]);
    x.copy_from(&t(&[4.0, 5.0, 6.0])).unwrap();
    assert_eq!(x.to_vec(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn mutation_bumps_the_shared_version_once_per_op() {
    let x = t(&[1.0, 2.0]);
    assert_eq!(x._version(), 0);
    x.add_scalar_(1.0).unwrap();
    assert_eq!(x._version(), 1);
    x.zero_().unwrap();
    assert_eq!(x._version(), 2);
    // A whole self-copy is a semantic no-op and adds no version noise.
    x.copy_from(&x).unwrap();
    assert_eq!(x._version(), 2);
}

#[test]
fn aliased_dst_and_src_is_the_self_op() {
    let x = t(&[1.0, -2.0, 3.0]);
    x.add_(&x).unwrap();
    assert_eq!(x.to_vec(), vec![2.0, -4.0, 6.0]);
    x.mul_(&x).unwrap();
    assert_eq!(x.to_vec(), vec![4.0, 16.0, 36.0]);
}

#[test]
fn mutation_is_visible_through_views_and_vice_versa() {
    // reshape of a contiguous base shares the whole storage: torch's
    // aliasing semantics, protected by the shared version counter.
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let flat = base.reshape(&[4]).unwrap();
    base.fill_(9.0).unwrap();
    assert_eq!(flat.to_vec(), vec![9.0; 4]);
    assert_eq!(base._version(), flat._version());

    flat.add_scalar_(1.0).unwrap();
    assert_eq!(base.to_vec(), vec![10.0; 4]);
}

#[test]
fn strided_source_is_materialized() {
    // dst must be whole-contiguous, but the SOURCE may be any f32 view:
    // a transpose view is non-contiguous and gets gathered first.
    let dst = Tensor::from_vec(vec![0.0; 4], &[2, 2]).unwrap();
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let src = base.transpose(0, 1).unwrap();
    dst.copy_from(&src).unwrap();
    assert_eq!(dst.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    dst.add_(&src).unwrap();
    assert_eq!(dst.to_vec(), vec![2.0, 6.0, 4.0, 8.0]);
}

#[test]
fn grad_requiring_and_history_carrying_tensors_are_refused() {
    let leaf = t(&[1.0, 2.0]).requires_grad_(true).unwrap();
    assert!(leaf.zero_().is_err(), "grad-requiring leaf");

    let x = t(&[1.0, 2.0]).requires_grad_(true).unwrap();
    let y = x.mul(&x).unwrap();
    assert!(y.fill_(0.0).is_err(), "tensor with op history");

    // The detached copy of that output is a fresh buffer: mutable.
    let snap = y.detach_copy();
    snap.fill_(0.0).unwrap();
    assert_eq!(snap.to_vec(), vec![0.0, 0.0]);
}

#[test]
fn non_whole_destinations_are_refused() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    // transpose views are non-contiguous. transpose() records autograd only
    // when grad is required, so this exercises the layout gate, not the
    // history gate.
    let tr = base.transpose(0, 1).unwrap();
    assert!(tr.fill_(0.0).is_err());
    assert_eq!(base.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn dtype_shape_and_scalar_gates() {
    let i = Tensor::from_vec_i64(vec![1, 2, 3], &[3]).unwrap();
    assert!(i.zero_().is_err(), "i64 destination");

    let x = t(&[1.0, 2.0]);
    assert!(x.add_(&t(&[1.0, 2.0, 3.0])).is_err(), "shape mismatch");
    assert!(
        x.copy_from(&Tensor::from_vec_i64(vec![1, 2], &[2]).unwrap())
            .is_err(),
        "i64 source"
    );
    // Failed calls leave values and version untouched.
    assert_eq!(x.to_vec(), vec![1.0, 2.0]);
    assert_eq!(x._version(), 0);
    assert_eq!(x.dtype(), DType::F32);
}

#[test]
fn mul_scalar_preserves_signed_zero_and_add_scalar_is_exact_identity() {
    // mul_scalar_ rides the shared affine kernel with add = -0.0 and
    // add_scalar_ with mul = 1.0; both identities are exact in IEEE f32.
    let x = t(&[-0.0, 0.0, -2.0]);
    x.mul_scalar_(1.0).unwrap();
    let v = x.to_vec();
    assert!(v[0] == 0.0 && v[0].is_sign_negative(), "-0.0 preserved");
    assert!(v[1] == 0.0 && v[1].is_sign_positive());
    let y = t(&[1.5, -3.25]);
    y.add_scalar_(0.0).unwrap();
    assert_eq!(y.to_vec(), vec![1.5, -3.25]);
}
