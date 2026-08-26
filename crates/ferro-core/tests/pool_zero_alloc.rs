//! Gate G5, host half (docs/CAPABILITY.md 4.2): with the buffer pool
//! installed, an MLP training step performs ZERO fresh host buffer
//! allocations for tensor storage after warmup - every f32 buffer the step
//! needs (op outputs, gradients, backward seeds, matmul scratch) comes back
//! out of the freelists the previous step's drops filled. The pool is
//! thread-local, so this test's counters see only this test's work even
//! under the parallel harness.

use ferro_core::nn::{Linear, Module, Relu, Sequential};
use ferro_core::optim::Adam;
use ferro_core::{pool, Rng, Tensor};

struct Setup {
    model: Sequential,
    opt: Adam,
    x: Tensor,
    y: Tensor,
}

fn setup() -> Setup {
    let rng = Rng::new(7);
    let model = Sequential::new(vec![
        Box::new(Linear::new(4, 8, &rng)),
        Box::new(Relu),
        Box::new(Linear::new(8, 1, &rng)),
    ]);
    let params: Vec<_> = model.named_parameters().into_iter().map(|(_, p)| p).collect();
    let opt = Adam::new(params, 0.01);
    let x = Tensor::randn(&[16, 4], &rng);
    let y = Tensor::randn(&[16, 1], &rng);
    Setup { model, opt, x, y }
}

fn step(s: &mut Setup) -> f32 {
    let pred = s.model.forward(&s.x).unwrap();
    let diff = pred.sub(&s.y).unwrap();
    let loss = diff.mul(&diff).unwrap().mean();
    s.opt.zero_grad();
    loss.backward();
    s.opt.step();
    loss.item()
}

#[test]
fn mlp_step_allocates_nothing_fresh_after_warmup() {
    pool::clear();
    pool::set_enabled(true);
    let mut s = setup();

    // Warmup: optimizer state allocates, the pool discovers every size
    // class this step's shapes need.
    let first = step(&mut s);
    for _ in 0..2 {
        step(&mut s);
    }

    let before = pool::stats();
    let mut last = first;
    for _ in 0..10 {
        last = step(&mut s);
    }
    let after = pool::stats();

    assert_eq!(
        after.misses - before.misses,
        0,
        "steady-state steps must draw every storage buffer from the pool \
         (fresh allocations after warmup: {})",
        after.misses - before.misses
    );
    let hits = after.hits - before.hits;
    assert!(hits >= 100, "the step really runs through the pool ({hits} hits)");
    assert!(
        last < first,
        "pooling must not perturb training: loss {first} -> {last}"
    );
}

#[test]
fn the_zero_miss_claim_has_teeth() {
    // Same loop with recycling disabled: every take misses. Proves the
    // counters actually observe this workload (the test above is not
    // vacuously green).
    pool::clear();
    pool::set_enabled(false);
    let mut s = setup();
    for _ in 0..3 {
        step(&mut s);
    }
    let before = pool::stats();
    step(&mut s);
    let after = pool::stats();
    assert!(
        after.misses - before.misses >= 20,
        "with the pool off, one step performs many fresh allocations (saw {})",
        after.misses - before.misses
    );
    pool::set_enabled(true);
    pool::clear();
}

#[test]
fn recycled_numerics_match_a_pool_free_run() {
    // The pool must be numerically invisible: identical seeds and shapes,
    // pool on vs off, bitwise-equal losses at every step.
    let run = |enabled: bool| -> Vec<u32> {
        pool::clear();
        pool::set_enabled(enabled);
        let mut s = setup();
        let losses = (0..8).map(|_| step(&mut s).to_bits()).collect();
        pool::set_enabled(true);
        pool::clear();
        losses
    };
    assert_eq!(run(true), run(false));
}
