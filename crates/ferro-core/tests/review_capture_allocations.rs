use ferro_core::{capture, Tensor};
use ferro_core::graph::{CompiledChain, Graph};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;

struct CountAlloc;
thread_local! {
    static WATCH: Cell<bool> = const { Cell::new(false) };
    static COPIES: Cell<usize> = const { Cell::new(0) };
}
const N: usize = 4093;
static LOCK: Mutex<()> = Mutex::new(());
#[global_allocator]
static ALLOC: CountAlloc = CountAlloc;

unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() == N * std::mem::size_of::<f32>() {
            let _ = WATCH.try_with(|watch| {
                if watch.get() { let _ = COPIES.try_with(|n| n.set(n.get() + 1)); }
            });
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn measured(f: impl FnOnce() -> Tensor) -> (Tensor, usize) {
    struct Reset;
    impl Drop for Reset { fn drop(&mut self) { WATCH.with(|w| w.set(false)); } }
    COPIES.with(|n| n.set(0));
    WATCH.with(|w| w.set(true));
    let reset = Reset;
    let out = f();
    drop(reset);
    (out, COPIES.with(Cell::get))
}

fn check(op: fn(&Tensor) -> Tensor) {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let x = Tensor::full(&[N], 0.75);
    // Warm backend initialization before measuring activation-sized allocations.
    drop(op(&x));
    let (eager, eager_copies) = measured(|| op(&x));
    assert_eq!(Graph::from_root(&eager).nodes.len(), 1);
    drop(eager);
    let (y, capture_copies) = measured(|| capture(|| op(&x)));
    assert_eq!(capture_copies, eager_copies, "capture allocated an extra activation snapshot");
    assert!(!y.requires_grad());
    let graph = Graph::from_root(&y);
    assert_eq!(graph.nodes.len(), 2);
    assert!(graph.nodes[&y.id()].tag.is_some());
    let compiled = CompiledChain::compile(&y).unwrap();
    x.fill_(1.25).unwrap();
    let expected = op(&x).to_vec();
    for (a, b) in compiled.replay().unwrap().to_vec().iter().zip(expected) {
        assert!((a - b).abs() < 1e-6);
    }
    for recording in [false, true] {
        let leaf = Tensor::from_vec(vec![0.4, 0.8, 1.3], &[3]).unwrap();
        ferro_core::testkit::grad_check(&[leaf], |xs| {
            if recording { capture(|| op(&xs[0]).sum()) } else { op(&xs[0]).sum() }
        });
    }
}

#[test]
fn gelu_capture_has_no_snapshot() { check(Tensor::gelu); }
#[test]
fn gelu_erf_capture_has_no_snapshot() { check(Tensor::gelu_erf); }
#[test]
fn sqrt_capture_has_no_snapshot() { check(Tensor::sqrt); }
#[test]
fn tanh_capture_has_no_snapshot() { check(Tensor::tanh); }
#[test]
fn exp_capture_has_no_snapshot() { check(Tensor::exp); }
#[test]
fn sigmoid_capture_has_no_snapshot() { check(Tensor::sigmoid); }
#[test]
fn silu_capture_has_no_snapshot() { check(Tensor::silu); }

fn check_normalization(log: bool) {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let op = |x: &Tensor| if log { x.log_softmax(0).unwrap() } else { x.softmax(0).unwrap() };
    let x = Tensor::full(&[N], 0.75);
    drop(op(&x));
    let (eager, eager_copies) = measured(|| op(&x));
    assert_eq!(Graph::from_root(&eager).nodes.len(), 1);
    drop(eager);
    let (y, capture_copies) = measured(|| capture(|| op(&x)));
    assert_eq!(capture_copies, eager_copies, "capture allocated an extra normalization snapshot");
    assert!(!y.requires_grad());
    assert_eq!(Graph::from_root(&y).nodes.len(), 2);
    if log {
        // LogSoftmax has no forward metadata: capture must retain the barrier.
        assert!(CompiledChain::compile(&y).is_err());
        assert!(CompiledChain::compile(&capture(|| y.neg())).is_err());
    } else {
        let compiled = CompiledChain::compile(&y).unwrap();
        let changed = Tensor::from_vec((0..N).map(|i| (i % 7) as f32 * 0.4).collect(), &[N]).unwrap();
        x.copy_from(&changed).unwrap();
        let replay = capture(|| compiled.replay().unwrap());
        assert_ne!(replay.to_vec(), y.to_vec());
        assert_eq!(Graph::from_root(&replay).nodes.len(), 1);
        for (a, b) in replay.to_vec().iter().zip(op(&x).to_vec()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    // A weighted loss avoids the constant sum(softmax) and tests both axes.
    let weights = Tensor::from_vec(vec![0.2, -0.7, 1.1, 0.8, -0.3, 0.5], &[2, 3]).unwrap();
    for dim in [0, 1] {
        for recording in [false, true] {
            let leaf = Tensor::from_vec(vec![0.4, -0.8, 1.3, -0.2, 0.7, 0.1], &[2, 3]).unwrap();
            ferro_core::testkit::grad_check(&[leaf], |xs| {
                let loss = || {
                    let y = if log { xs[0].log_softmax(dim).unwrap() } else { xs[0].softmax(dim).unwrap() };
                    y.mul(&weights).unwrap().sum()
                };
                if recording { capture(loss) } else { loss() }
            });
        }
    }
}

#[test]
fn softmax_capture_has_no_snapshot() { check_normalization(false); }
#[test]
fn log_softmax_capture_has_no_snapshot() { check_normalization(true); }
