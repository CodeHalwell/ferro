//! Ferro-owned caching device allocator for f32 buffers.
//!
//! ## Why this exists
//! `cudarc` (0.19.x) already routes `CudaStream::alloc*` through the driver's
//! stream-ordered memory pool (`cudaMallocAsync`) on devices that support it
//! (the RTX 3090 does), so ferro is *not* doing naive `cudaMalloc`/`cudaFree`
//! per op. What the driver pool does NOT give us is (a) elimination of the
//! per-allocation `malloc_async`/`free_async` *call* + bookkeeping cost, and
//! (b) a **deterministic, in-process count** of how many fresh device
//! allocations a training step actually requests.
//!
//! Gate G6 is a *structural* proof — "zero fresh device buffer requests per
//! step after warm-up" — and you can only prove that honestly if ferro owns
//! the freelist and counts its own misses, rather than asserting something
//! about the opaque driver pool. This allocator is also the prerequisite for
//! static memory planning and stable buffer addresses across steps, which is
//! what unblocks full-step CUDA graph capture.
//!
//! ## Design
//! Size-binned freelists keyed by exact element length. A request for `len`
//! f32s first checks the bin for `len`; a hit pops a recycled `CudaSlice`
//! (zeroed before hand-back so callers keep `alloc_zeros` semantics), a miss
//! calls the driver exactly once and increments `driver_allocs`. On drop, a
//! buffer's slice is pushed back into its bin instead of being freed, capped
//! per bin so long-running processes don't hoard. Exact-length binning is the
//! honest choice for the G6 proof: ferro's graph re-requests the *same* shapes
//! every step, so exact bins recycle at 100% after warm-up without the
//! rounding waste of power-of-two bins.

use cudarc::driver::{CudaSlice, CudaStream};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Max recycled slices retained per size bin. Bounds resident pool growth for
/// shapes that are allocated in bursts then never reused at that multiplicity.
const MAX_PER_BIN: usize = 64;

/// Counters for structural proofs (G6). All monotonically increasing over the
/// allocator's lifetime; take a [`AllocStats`] snapshot before/after a step and
/// diff to get per-step figures.
#[derive(Default)]
struct Counters {
    /// Total buffer requests routed through the allocator.
    requests: AtomicU64,
    /// Requests served from a freelist bin (no driver call).
    hits: AtomicU64,
    /// Requests that fell through to a fresh driver allocation.
    driver_allocs: AtomicU64,
    /// Slices returned to a bin on drop.
    recycled: AtomicU64,
    /// Slices dropped to the driver because their bin was full.
    released: AtomicU64,
}

/// Immutable snapshot of allocator counters at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocStats {
    pub requests: u64,
    pub hits: u64,
    pub driver_allocs: u64,
    pub recycled: u64,
    pub released: u64,
}

impl AllocStats {
    /// Per-interval delta: `self` (later) minus `earlier`.
    pub fn since(&self, earlier: &AllocStats) -> AllocStats {
        AllocStats {
            requests: self.requests - earlier.requests,
            hits: self.hits - earlier.hits,
            driver_allocs: self.driver_allocs - earlier.driver_allocs,
            recycled: self.recycled - earlier.recycled,
            released: self.released - earlier.released,
        }
    }
}

/// Shared caching allocator. Cheap to clone (`Arc` inside); every [`CudaBuf`]
/// carries a clone so it can return its slice on drop.
#[derive(Clone)]
pub struct CachingAllocator {
    inner: Arc<AllocInner>,
}

struct AllocInner {
    stream: Arc<CudaStream>,
    bins: Mutex<HashMap<usize, Vec<CudaSlice<f32>>>>,
    counters: Counters,
    /// When false, the allocator is a pure pass-through to the driver (no
    /// recycling). Lets the G6 benchmark measure allocator-on vs allocator-off
    /// on the identical code path.
    enabled: bool,
}

/// Lock a mutex, recovering the guard even if a previous holder panicked while
/// holding it. The allocator is locked from `CudaBuf::drop`, so a plain
/// `.unwrap()` would turn one poisoned lock into an abort cascade on every
/// subsequent buffer drop; the freelist invariants hold regardless of who
/// panicked, so recovering the inner guard is safe and strictly better.
fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl CachingAllocator {
    /// Recycling allocator bound to `stream`.
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self::with_enabled(stream, true)
    }

    /// Pass-through allocator (no freelist) — every request hits the driver.
    /// Used as the G6 baseline arm.
    pub fn passthrough(stream: Arc<CudaStream>) -> Self {
        Self::with_enabled(stream, false)
    }

    fn with_enabled(stream: Arc<CudaStream>, enabled: bool) -> Self {
        CachingAllocator {
            inner: Arc::new(AllocInner {
                stream,
                bins: Mutex::new(HashMap::new()),
                counters: Counters::default(),
                enabled,
            }),
        }
    }

    /// Allocate `len` zeroed f32s, serving from the freelist when possible.
    /// Mirrors `CudaStream::alloc_zeros` semantics (result is zeroed).
    pub fn alloc_zeros(&self, len: usize) -> Result<CudaSlice<f32>, cudarc::driver::DriverError> {
        let c = &self.inner.counters;
        c.requests.fetch_add(1, Ordering::Relaxed);

        if self.inner.enabled {
            // Try the freelist. len==0 slices are legal and binned like any
            // other; recycling them keeps the count honest.
            let recycled = {
                let mut bins = lock_recover(&self.inner.bins);
                bins.get_mut(&len).and_then(|v| v.pop())
            };
            if let Some(mut slice) = recycled {
                c.hits.fetch_add(1, Ordering::Relaxed);
                // Re-zero to preserve alloc_zeros semantics for the caller.
                self.inner.stream.memset_zeros(&mut slice)?;
                return Ok(slice);
            }
        }

        c.driver_allocs.fetch_add(1, Ordering::Relaxed);
        match self.inner.stream.alloc_zeros::<f32>(len) {
            Ok(s) => Ok(s),
            Err(_) if self.inner.enabled => {
                // Flush the pool and retry once (see alloc_uninit note).
                self.clear_pool();
                self.inner.stream.alloc_zeros::<f32>(len)
            }
            Err(e) => Err(e),
        }
    }

    /// Allocate `len` UNINITIALISED f32s, serving from the freelist when
    /// possible. Unlike [`alloc_zeros`], a freelist hit is handed back WITHOUT
    /// re-zeroing and a miss uses `alloc` (no memset), so this is the fast path
    /// for buffers the caller fully overwrites (matmul with beta=0, elementwise
    /// maps, gather). Using it for a buffer that is only partially written
    /// would expose stale bytes — callers must guarantee a full overwrite.
    ///
    /// # Safety
    /// The returned slice may contain arbitrary prior contents; the caller must
    /// write every element it later reads.
    pub unsafe fn alloc_uninit(
        &self,
        len: usize,
    ) -> Result<CudaSlice<f32>, cudarc::driver::DriverError> {
        let c = &self.inner.counters;
        c.requests.fetch_add(1, Ordering::Relaxed);

        if self.inner.enabled {
            let recycled = {
                let mut bins = lock_recover(&self.inner.bins);
                bins.get_mut(&len).and_then(|v| v.pop())
            };
            if let Some(slice) = recycled {
                c.hits.fetch_add(1, Ordering::Relaxed);
                // No re-zero: caller overwrites the whole buffer.
                return Ok(slice);
            }
        }

        c.driver_allocs.fetch_add(1, Ordering::Relaxed);
        match self.inner.stream.alloc::<f32>(len) {
            Ok(s) => Ok(s),
            Err(_) if self.inner.enabled => {
                self.clear_pool();
                self.inner.stream.alloc::<f32>(len)
            }
            Err(e) => Err(e),
        }
    }

    /// Return a slice to its size bin (called from `CudaBuf::drop`). Slices are
    /// only recycled when the allocator is enabled and the bin isn't full;
    /// otherwise the slice drops here and the driver reclaims it.
    pub fn recycle(&self, slice: CudaSlice<f32>) {
        if !self.inner.enabled {
            self.inner.counters.released.fetch_add(1, Ordering::Relaxed);
            return; // slice dropped -> driver free
        }
        let len = slice.len();
        let mut bins = lock_recover(&self.inner.bins);
        let bin = bins.entry(len).or_default();
        if bin.len() < MAX_PER_BIN {
            bin.push(slice);
            self.inner.counters.recycled.fetch_add(1, Ordering::Relaxed);
        } else {
            drop(bins);
            self.inner.counters.released.fetch_add(1, Ordering::Relaxed);
            // slice dropped here -> driver free
        }
    }

    /// Snapshot the counters for a structural proof.
    pub fn stats(&self) -> AllocStats {
        let c = &self.inner.counters;
        AllocStats {
            requests: c.requests.load(Ordering::Relaxed),
            hits: c.hits.load(Ordering::Relaxed),
            driver_allocs: c.driver_allocs.load(Ordering::Relaxed),
            recycled: c.recycled.load(Ordering::Relaxed),
            released: c.released.load(Ordering::Relaxed),
        }
    }

    /// Drop every pooled slice back to the driver. Frees the resident pool
    /// without tearing down the allocator handle.
    pub fn clear_pool(&self) {
        let mut bins = lock_recover(&self.inner.bins);
        bins.clear();
    }

    /// Total slices currently held in the freelist (all bins).
    pub fn pooled(&self) -> usize {
        lock_recover(&self.inner.bins).values().map(Vec::len).sum()
    }

    pub fn is_enabled(&self) -> bool {
        self.inner.enabled
    }
}
