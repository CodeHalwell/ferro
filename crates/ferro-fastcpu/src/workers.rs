//! Persistent worker pool for the elementwise/reduction kernels. A per-call
//! std::thread::scope pays a spawn+join for every worker on every op, which
//! dominated mid-sized tensors; these workers start once (lazily, sized to
//! available_parallelism - 1, so a taskset/cgroup-limited process gets the
//! cores it is allowed) and park on a condvar after a short spin.
//!
//! One job runs at a time. A caller that finds the pool busy (another thread's
//! job, or a nested call from inside a task) runs its tasks inline instead of
//! queueing, so the pool can never deadlock on itself. The caller waits for
//! its tasks, not for every worker: a worker that wakes late finds nothing
//! left to claim.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, TryLockError};

struct Job {
    // Lifetime-erased borrow of the caller's closure. Sound because `run`
    // does not return until `done == tasks`, and `f` is only called for a
    // claimed index < tasks, counted in `done` after the call returns. A
    // worker holding a finished job claims an index >= tasks and never
    // dereferences `f`.
    f: *const (dyn Fn(usize) + Sync),
    tasks: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    panicked: AtomicBool,
}

// SAFETY: `f` points at a Sync closure kept alive by the blocked caller.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    fn work(&self, p: &Pool) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.tasks {
                return;
            }
            // SAFETY: see the field comment on `f`.
            let f = unsafe { &*self.f };
            if catch_unwind(AssertUnwindSafe(|| f(i))).is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
            if self.done.fetch_add(1, Ordering::AcqRel) + 1 == self.tasks {
                let _g = p.done.lock().unwrap_or_else(|e| e.into_inner());
                p.done_cv.notify_all();
            }
        }
    }
}

struct Pool {
    workers: usize,
    epoch: AtomicU64,
    slot: Mutex<Option<Arc<Job>>>,
    wake: Condvar,
    done: Mutex<()>,
    done_cv: Condvar,
    gate: Mutex<()>,
}

/// Idle spin before parking (~0.1-0.3 ms): back-to-back ops skip the futex
/// wake without burning a core for long between bursts.
const SPIN: usize = 1 << 11;

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let workers = std::thread::available_parallelism().map_or(1, |p| p.get()) - 1;
        let p: &'static Pool = Box::leak(Box::new(Pool {
            workers,
            epoch: AtomicU64::new(0),
            slot: Mutex::new(None),
            wake: Condvar::new(),
            done: Mutex::new(()),
            done_cv: Condvar::new(),
            gate: Mutex::new(()),
        }));
        for w in 0..workers {
            std::thread::Builder::new().name(format!("ferro-fastcpu-{w}")).spawn(move || worker(p)).expect("spawn fastcpu worker");
        }
        p
    })
}

fn worker(p: &'static Pool) {
    let mut seen = 0;
    loop {
        let mut spins = 0;
        while p.epoch.load(Ordering::Acquire) == seen && spins < SPIN {
            std::hint::spin_loop();
            spins += 1;
        }
        let job = {
            let mut g = p.slot.lock().unwrap_or_else(|e| e.into_inner());
            while p.epoch.load(Ordering::Acquire) == seen {
                g = p.wake.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            seen = p.epoch.load(Ordering::Acquire);
            g.clone()
        };
        if let Some(job) = job {
            job.work(p);
        }
    }
}

/// Total threads a `run` can use (workers plus the caller).
pub fn threads() -> usize {
    pool().workers + 1
}

/// Runs `f(0..tasks)` across the pool, the caller included; returns once every
/// task has finished. A panicking task is re-raised here after all finish.
pub fn run(tasks: usize, f: &(dyn Fn(usize) + Sync)) {
    let p = pool();
    let gate = match p.gate.try_lock() {
        Ok(g) => Some(g),
        Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    };
    if tasks <= 1 || p.workers == 0 || gate.is_none() {
        (0..tasks).for_each(f);
        return;
    }
    // SAFETY: erases the borrow's lifetime only; see `Job::f`.
    let f: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
    let job = Arc::new(Job { f, tasks, next: AtomicUsize::new(0), done: AtomicUsize::new(0), panicked: AtomicBool::new(false) });
    {
        let mut g = p.slot.lock().unwrap_or_else(|e| e.into_inner());
        *g = Some(job.clone());
        p.epoch.fetch_add(1, Ordering::Release);
        p.wake.notify_all();
    }
    job.work(p);
    let mut g = p.done.lock().unwrap_or_else(|e| e.into_inner());
    while job.done.load(Ordering::Acquire) != tasks {
        g = p.done_cv.wait(g).unwrap_or_else(|e| e.into_inner());
    }
    drop(g);
    drop(gate);
    assert!(!job.panicked.load(Ordering::Relaxed), "a fastcpu worker task panicked");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_every_task_once_and_nests_inline() {
        let hits: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
        run(hits.len(), &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
            let inner = AtomicUsize::new(0);
            run(3, &|_| {
                inner.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(inner.load(Ordering::Relaxed), 3);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn concurrent_callers_and_panics_are_contained() {
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..200 {
                        let n = AtomicUsize::new(0);
                        run(64, &|_| {
                            n.fetch_add(1, Ordering::Relaxed);
                        });
                        assert_eq!(n.load(Ordering::Relaxed), 64);
                    }
                });
            }
        });
        let r = catch_unwind(|| run(16, &|i| assert!(i != 7, "intentional task panic")));
        assert!(r.is_err());
        let n = AtomicUsize::new(0);
        run(16, &|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 16);
    }
}
