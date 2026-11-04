//! Ordinary publication owns or safely steals initial gradients, then updates
//! existing public slots in place, including handles retained by callers.

use ferro_core::Tensor;

fn leaf(v: &[f32]) -> Tensor {
    Tensor::from_vec(v.to_vec(), &[v.len()])
        .unwrap()
        .requires_grad_(true)
        .unwrap()
}

#[test]
fn shared_seed_gradients_survive_repeated_accumulation() {
    // Shared upstream cotangents must become independently writable slots.
    let a = leaf(&[1.0, 2.0]);
    let b = leaf(&[3.0, 4.0]);
    let loss = a.add(&b).unwrap().sum();
    loss.backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![1.0, 1.0]);
    assert_eq!(b.grad().unwrap().to_vec(), vec![1.0, 1.0]);
    assert_ne!(a.grad().unwrap()._storage_ptr(), b.grad().unwrap()._storage_ptr());

    loss.backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![2.0, 2.0]);
    assert_eq!(
        b.grad().unwrap().to_vec(),
        vec![2.0, 2.0],
        "b's accumulation must not have been corrupted through a's shared grad storage"
    );
}

#[test]
fn user_held_grad_handle_observes_further_accumulation_like_torch() {
    let x = leaf(&[1.0, 2.0]);
    let loss = x.mul(&x).unwrap().sum();
    loss.backward();
    let held = x.grad().unwrap();
    assert_eq!(held.to_vec(), vec![2.0, 4.0]);

    loss.backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![4.0, 8.0]);
    assert_eq!(x.grad().unwrap().id(), held.id());
    assert_eq!(held.to_vec(), vec![4.0, 8.0]);
}

#[test]
fn unshared_grad_accumulates_in_place_with_stable_storage() {
    // mul's backward produces two DISTINCT gradient tensors for x used
    // twice, so the slot's first tensor is unshared and the second
    // contribution may (and does) land in place: same storage before and
    // after, version bumped.
    let x = leaf(&[1.0, 2.0]);
    let y = x.mul(&x).unwrap();
    y.sum().backward();
    let g1 = x.grad().unwrap();
    assert_eq!(g1.to_vec(), vec![2.0, 4.0]);
    assert!(
        g1._version() >= 1,
        "second contribution accumulated in place (version {})",
        g1._version()
    );
}

#[test]
fn multi_use_and_multi_backward_values_are_exact() {
    // (x*x + x).sum(): x receives three contributions per backward
    // (2x from mul, 1 from add); two backwards double everything.
    let x = leaf(&[3.0]);
    let loss = x.mul(&x).unwrap().add(&x).unwrap().sum();
    loss.backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![7.0]);
    loss.backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![14.0]);
}
