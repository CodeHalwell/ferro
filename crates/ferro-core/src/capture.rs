//! Scoped inference graph recording through the ordinary op recording hooks.
use std::cell::Cell;

thread_local! {
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn is_recording() -> bool {
    DEPTH.with(|depth| depth.get() != 0)
}

pub(crate) fn without_recording<T>(run: impl FnOnce() -> T) -> T {
    struct Guard(usize);
    impl Drop for Guard {
        fn drop(&mut self) { DEPTH.with(|depth| depth.set(self.0)); }
    }
    let _guard = Guard(DEPTH.with(|depth| depth.replace(0)));
    run()
}

/// Record op inputs and kernel tags while `build` executes, without changing
/// requires_grad. Nested scopes are supported and panic unwinding restores the
/// calling thread's state. Other threads are unaffected. The returned tensor
/// retains its recorded inputs; compile it with `graph::CompiledChain::compile`.
/// Capture executes eagerly; only operations supported by the compiler replay.
/// Existing gradient-requiring inputs retain their ordinary autograd behavior.
/// Inference-only f32 device transpose/reshape outputs own their eager storage
/// when they would otherwise alias a graph leaf. This capture-time copy lets
/// retained tapes coexist with public input updates; it does not permit updates
/// through ordinary device aliases or autograd snapshots. Replay is not captured
/// and does not pay this copy. Host views retain their ordinary alias semantics.
pub fn capture<T>(build: impl FnOnce() -> T) -> T {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            DEPTH.with(|depth| depth.set(depth.get() - 1));
        }
    }
    DEPTH.with(|depth| depth.set(depth.get() + 1));
    let _guard = Guard;
    build()
}
