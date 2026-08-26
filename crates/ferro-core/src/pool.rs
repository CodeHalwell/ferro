//! Host buffer pool: thread-local, exact-size freelists recycling the
//! `Vec<f32>` behind f32 tensor storage (docs/CAPABILITY.md 4.2). Fixed-shape
//! training reuses the same buffer sizes every step, so `StorageCell::drop`
//! gives storage back here and the kernels/constructors take it out again -
//! after warmup a step performs zero fresh host allocations for tensor
//! storage (tests/pool_zero_alloc.rs counts it, residency-test style).
//! Beyond allocator cost this dodges page zeroing: a fresh `vec![0f32; n]`
//! page-faults per 4 KB on first touch, a recycled buffer is warm.
//!
//! Take flavors and their contracts:
//! - `take_uninit(n)`: contents are arbitrary (but valid) f32 values from the
//!   buffer's previous life; the caller MUST write every element before any
//!   read. Debug builds poison recycled contents with NaN, so a kernel that
//!   misses a slot turns the whole test suite into the audit.
//! - `take_zeroed(n)` / `take_filled(n, v)`: recycled contents are cleared,
//!   safe for accumulator-style kernels (matmul += into its output).
//! - `give(v)`: recycle a buffer. Called by `StorageCell::drop` for every
//!   f32 storage, and explicitly by the few internal sites that take pooled
//!   temporaries (keeping takes and gives balanced by construction).
//!
//! Thread-local by design (the "thread-local fast path"): no locks, and each
//! test thread sees only its own stats, so structural assertions stay
//! deterministic under the parallel harness. A buffer allocated on one
//! thread and dropped on another simply migrates to the dropper's pool.
//! All access goes through `try_with`, so drops during thread teardown (when
//! the TLS slot is already destroyed) degrade to plain frees.

use std::cell::RefCell;
use std::collections::HashMap;

/// At most this many buffers are kept per size class; further gives of that
/// size are dropped. Fixed shapes need one buffer per simultaneously-live
/// tensor of that size, which a training step keeps well under this.
const PER_CLASS_CAP: usize = 16;

/// Default total bytes the pool may hold before gives start dropping.
const DEFAULT_CAP_BYTES: usize = 256 << 20;

/// Counters since thread start (or `clear`); deltas around a region prove
/// structural claims: `misses` unchanged means every storage buffer the
/// region needed came out of the freelists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// take_* calls served from a freelist.
    pub hits: usize,
    /// take_* calls that had to allocate fresh.
    pub misses: usize,
    /// give calls that entered a freelist.
    pub recycled: usize,
    /// give calls dropped instead (pool disabled, class full, or over cap).
    pub dropped: usize,
    /// Bytes currently held across all freelists.
    pub held_bytes: usize,
}

struct HostPool {
    classes: HashMap<usize, Vec<Vec<f32>>>,
    enabled: bool,
    cap_bytes: usize,
    stats: PoolStats,
}

impl HostPool {
    fn new() -> HostPool {
        HostPool {
            classes: HashMap::new(),
            enabled: true,
            cap_bytes: DEFAULT_CAP_BYTES,
            stats: PoolStats::default(),
        }
    }

    fn take(&mut self, n: usize) -> Option<Vec<f32>> {
        if !self.enabled {
            self.stats.misses += 1;
            return None;
        }
        match self.classes.get_mut(&n).and_then(|c| c.pop()) {
            Some(v) => {
                debug_assert_eq!(v.len(), n);
                self.stats.hits += 1;
                self.stats.held_bytes -= n * 4;
                Some(v)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    fn give(&mut self, v: Vec<f32>) {
        let n = v.len();
        if !self.enabled || n == 0 || self.stats.held_bytes + n * 4 > self.cap_bytes {
            self.stats.dropped += 1;
            return;
        }
        let class = self.classes.entry(n).or_default();
        if class.len() >= PER_CLASS_CAP {
            self.stats.dropped += 1;
            return;
        }
        class.push(v);
        self.stats.recycled += 1;
        self.stats.held_bytes += n * 4;
    }
}

thread_local! {
    static POOL: RefCell<HostPool> = RefCell::new(HostPool::new());
}

/// A recycled (or fresh) buffer of exactly `n` elements whose CONTENTS ARE
/// ARBITRARY f32 values: the caller must write every element before reading
/// any. Debug builds poison recycled contents with NaN so a violation
/// propagates loudly through the numerics instead of reproducing stale data.
pub fn take_uninit(n: usize) -> Vec<f32> {
    let recycled = POOL.try_with(|p| p.borrow_mut().take(n)).ok().flatten();
    match recycled {
        #[allow(unused_mut)]
        Some(mut v) => {
            #[cfg(debug_assertions)]
            v.fill(f32::NAN);
            v
        }
        // Fresh zeroed allocation: vec! is the cheapest correct fresh path
        // (calloc/mmap), and fresh memory poisoning would defeat it.
        None => vec![0f32; n],
    }
}

/// A buffer of `n` zeros (recycled contents are cleared).
pub fn take_zeroed(n: usize) -> Vec<f32> {
    take_filled(n, 0.0)
}

/// A buffer of `n` copies of `value`.
pub fn take_filled(n: usize, value: f32) -> Vec<f32> {
    let recycled = POOL.try_with(|p| p.borrow_mut().take(n)).ok().flatten();
    match recycled {
        Some(mut v) => {
            v.fill(value);
            v
        }
        None => vec![value; n],
    }
}

/// Recycle a buffer into this thread's pool (dropped instead when the pool
/// is disabled, the size class is full, the byte cap is reached, or the
/// thread is tearing down).
pub fn give(v: Vec<f32>) {
    if v.is_empty() {
        return;
    }
    let _ = POOL.try_with(|p| p.borrow_mut().give(v));
}

/// This thread's counters since thread start (or the last `clear`).
pub fn stats() -> PoolStats {
    POOL.try_with(|p| p.borrow().stats).unwrap_or_default()
}

/// Drop every held buffer and reset this thread's counters.
pub fn clear() {
    let _ = POOL.try_with(|p| {
        let mut p = p.borrow_mut();
        p.classes.clear();
        p.stats = PoolStats::default();
    });
}

/// Turn recycling on or off for this thread (off: takes allocate fresh,
/// gives drop). Useful for A/B benchmarking; the pool starts enabled.
pub fn set_enabled(on: bool) {
    let _ = POOL.try_with(|p| p.borrow_mut().enabled = on);
}

/// Cap the bytes this thread's pool may hold (default 256 MiB).
pub fn set_cap_bytes(cap: usize) {
    let _ = POOL.try_with(|p| p.borrow_mut().cap_bytes = cap);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each test drives its own thread-local pool; clear() gives a clean
    /// slate even when the harness reuses a thread.
    fn fresh() {
        clear();
        set_enabled(true);
    }

    #[test]
    fn take_after_give_reuses_the_allocation() {
        fresh();
        let mut v = take_zeroed(1024);
        let ptr = v.as_ptr();
        v[0] = 42.0;
        give(v);
        let v2 = take_zeroed(1024);
        assert_eq!(v2.as_ptr(), ptr, "same allocation came back");
        assert_eq!(v2[0], 0.0, "recycled contents were cleared");
        let s = stats();
        assert_eq!((s.hits, s.misses, s.recycled), (1, 1, 1));
    }

    #[test]
    fn classes_are_exact_size() {
        fresh();
        give(take_zeroed(8));
        let v = take_zeroed(9);
        assert_eq!(v.len(), 9);
        assert_eq!(stats().hits, 0, "a 9-take must not consume the 8-class");
    }

    #[test]
    fn take_filled_overwrites_recycled_garbage() {
        fresh();
        let mut v = take_uninit(16);
        v.fill(7.5);
        give(v);
        assert!(take_filled(16, 2.0).iter().all(|&x| x == 2.0));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn debug_take_uninit_poisons_recycled_contents() {
        fresh();
        let mut v = take_uninit(4);
        v.fill(1.0);
        give(v);
        assert!(take_uninit(4).iter().all(|x| x.is_nan()));
    }

    #[test]
    fn per_class_cap_and_byte_cap_drop_excess() {
        fresh();
        for _ in 0..PER_CLASS_CAP + 3 {
            give(vec![0.0; 4]);
        }
        let s = stats();
        assert_eq!(s.recycled, PER_CLASS_CAP);
        assert_eq!(s.dropped, 3);

        fresh();
        set_cap_bytes(4 * 10);
        give(vec![0.0; 8]); // 32 bytes: fits
        give(vec![0.0; 8]); // would exceed 40 bytes: dropped
        let s = stats();
        assert_eq!((s.recycled, s.dropped), (1, 1));
        set_cap_bytes(DEFAULT_CAP_BYTES);
    }

    #[test]
    fn disabled_pool_allocates_and_drops() {
        fresh();
        set_enabled(false);
        give(vec![0.0; 32]);
        let _ = take_zeroed(32);
        let s = stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.dropped, 1);
        set_enabled(true);
    }
}
