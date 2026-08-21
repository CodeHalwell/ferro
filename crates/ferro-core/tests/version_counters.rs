use ferro_core::Tensor;

#[test]
fn view_and_base_share_version() {
    // A view created via transpose/reshape shares the same Arc<StorageCell> as
    // its base, so both must report the same version, and a bump through
    // either side is visible from the other.
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let view = base.transpose(0, 1).unwrap();
    assert_eq!(base._version(), view._version());

    view._bump_version_for_test();
    assert_eq!(base._version(), view._version());
    assert_eq!(base._version(), 1);

    let reshaped = base.reshape(&[6]).unwrap();
    assert_eq!(reshaped._version(), base._version());
    base._bump_version_for_test();
    assert_eq!(reshaped._version(), base._version());
}

#[test]
fn detach_copy_has_independent_version() {
    // detach_copy of a host tensor allocates a fresh Vec, hence a fresh
    // StorageCell: bumping the copy must not affect the source, and vice
    // versa, so a backward through the original is never poisoned by activity
    // on a detached snapshot (the immunity the Op doc comment describes).
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let copy = x.detach_copy();
    assert_eq!(x._version(), 0);
    assert_eq!(copy._version(), 0);

    copy._bump_version_for_test();
    copy._bump_version_for_test();
    assert_eq!(copy._version(), 2);
    assert_eq!(x._version(), 0, "bumping a detached copy must not touch the source");

    x._bump_version_for_test();
    assert_eq!(x._version(), 1);
    assert_eq!(copy._version(), 2, "copy's counter is unaffected by the source's bump too");

    // Prove it end to end: a mul recorded against x backward()s fine after the
    // copy (not x) was mutated.
    let x = Tensor::from_vec(vec![2.0, -1.0], &[2]).unwrap().requires_grad_(true);
    let y = x.mul(&x).unwrap();
    let snapshot = y.detach_copy();
    snapshot._bump_version_for_test();
    y.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![4.0, -2.0]);
}

#[test]
#[should_panic(expected = "one of the variables needed for gradient computation has been modified by an inplace operation")]
fn mutated_saved_input_panics_on_backward() {
    // mul saves both operands for its backward (d/dx(x*x) = 2x needs x); bump
    // x's version after recording and backward must refuse to trust it.
    let x = Tensor::from_vec(vec![3.0, -2.0], &[2]).unwrap().requires_grad_(true);
    let y = x.mul(&x).unwrap();
    x._bump_version_for_test();
    y.sum().backward();
}

#[test]
fn repeated_backward_snapshots_are_per_record_not_per_backward() {
    // Version snapshots are taken once, at record_fn time, not refreshed on
    // every backward() call - so retain-graph-style repeated backward through
    // an untouched graph must stay green across multiple calls.
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap().requires_grad_(true);
    let loss = x.mul(&x).unwrap().sum();
    loss.backward();
    loss.backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![4.0, 8.0, 12.0]);
}
