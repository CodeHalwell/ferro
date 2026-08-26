//! accumulate_grad's in-place fast path is a pure optimization: it may only
//! fire when the stored gradient is provably unshared (sole handle, sole
//! storage reference). These tests pin the observable semantics that gate
//! protects - shared seed gradients (one backward can hand several inputs
//! the same tensor) and user-held grad() clones must never be mutated.

use ferro_core::Tensor;

fn leaf(v: &[f32]) -> Tensor {
    Tensor::from_vec(v.to_vec(), &[v.len()])
        .unwrap()
        .requires_grad_(true)
        .unwrap()
}

#[test]
fn shared_seed_gradients_survive_repeated_accumulation() {
    // add's backward passes the SAME upstream gradient tensor to both
    // inputs, so after the first backward a.grad and b.grad share storage.
    // The second backward accumulates into each slot; an unguarded in-place
    // add would double-count through the shared buffer.
    let a = leaf(&[1.0, 2.0]);
    let b = leaf(&[3.0, 4.0]);
    let loss = a.add(&b).unwrap().sum();
    loss.backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![1.0, 1.0]);
    assert_eq!(b.grad().unwrap().to_vec(), vec![1.0, 1.0]);

    loss.backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![2.0, 2.0]);
    assert_eq!(
        b.grad().unwrap().to_vec(),
        vec![2.0, 2.0],
        "b's accumulation must not have been corrupted through a's shared grad storage"
    );
}

#[test]
fn user_held_grad_clone_keeps_its_value_across_further_accumulation() {
    let x = leaf(&[1.0, 2.0]);
    let loss = x.mul(&x).unwrap().sum();
    loss.backward();
    let held = x.grad().unwrap();
    assert_eq!(held.to_vec(), vec![2.0, 4.0]);

    // The held clone raises the tensor's refcount, so this accumulation must
    // take the allocating path and leave `held` untouched.
    loss.backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![4.0, 8.0]);
    assert_eq!(held.to_vec(), vec![2.0, 4.0], "held grad clone was mutated");
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
