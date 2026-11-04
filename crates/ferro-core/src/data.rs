//! Data loading: datasets, samplers, collate functions and a DataLoader with
//! std::thread-based prefetch workers. ferro-core stays zero-dependency, so
//! worker parallelism uses plain threads plus mpsc channels.

use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::tensor::Tensor;
use std::collections::BTreeMap;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

/// An (input, target) pairs source addressable by index.
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Result<(Tensor, Tensor)>;
}

/// In-memory dataset backed by two aligned leading-batch tensors.
pub struct TensorDataset {
    xs: Tensor,
    ys: Tensor,
}

impl TensorDataset {
    pub fn new(xs: Tensor, ys: Tensor) -> Result<Self> {
        let op = "TensorDataset";
        if xs.ndim() == 0 || ys.ndim() == 0 {
            return Err(Error::InvalidShape {
                op,
                msg: "inputs and targets need a leading batch dimension".into(),
            });
        }
        if xs.shape()[0] != ys.shape()[0] {
            return Err(Error::InvalidShape {
                op,
                msg: format!(
                    "input/target batch sizes differ: {} vs {}",
                    xs.shape()[0],
                    ys.shape()[0]
                ),
            });
        }
        Ok(TensorDataset { xs, ys })
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.xs.shape()[0]
    }

    fn get(&self, idx: usize) -> Result<(Tensor, Tensor)> {
        if idx >= self.len() {
            return Err(Error::InvalidShape {
                op: "TensorDataset",
                msg: format!("index {idx} out of range"),
            });
        }
        let x = self
            .xs
            .index_select(0, &[idx])?
            .reshape(&self.xs.shape()[1..])?;
        let y = self
            .ys
            .index_select(0, &[idx])?
            .reshape(&self.ys.shape()[1..])?;
        Ok((x, y))
    }
}

/// Produces the visit order over one epoch of `n` examples.
pub trait Sampler: Send + Sync {
    fn order(&self, n: usize) -> Vec<usize>;
}

pub struct SequentialSampler;

impl Sampler for SequentialSampler {
    fn order(&self, n: usize) -> Vec<usize> {
        (0..n).collect()
    }
}

/// Fisher-Yates shuffle seeded from ferro-core's own xorshift128+ rng.
pub struct RandomSampler {
    pub seed: u64,
}

impl Sampler for RandomSampler {
    fn order(&self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        let rng = Rng::new(self.seed);
        for i in (1..n).rev() {
            let j = (rng.uniform() * (i + 1) as f32).min(i as f32) as usize;
            idx.swap(i, j);
        }
        idx
    }
}

/// Combines a batch of items into stacked input/target tensors.
pub trait CollateFn: Send + Sync {
    fn collate(&self, items: &[(Tensor, Tensor)]) -> Result<(Tensor, Tensor)>;
}

/// Default collate: stacks items along a new leading dimension, preserving
/// each side's dtype. Items on the same side must have matching dtype/device.
pub struct StackCollate;

impl CollateFn for StackCollate {
    fn collate(&self, items: &[(Tensor, Tensor)]) -> Result<(Tensor, Tensor)> {
        let op = "collate";
        if items.is_empty() {
            return Err(Error::InvalidShape {
                op,
                msg: "cannot collate an empty batch".into(),
            });
        }
        let stack_side = |side: &[Tensor]| -> Result<Tensor> {
            let rows: Result<Vec<Tensor>> = side
                .iter()
                .map(|t| {
                    let mut s = t.shape().to_vec();
                    s.insert(0, 1);
                    t.reshape(&s)
                })
                .collect();
            Tensor::cat(&rows?, 0)
        };
        let (xs, ys): (Vec<_>, Vec<_>) = items.iter().map(|(x, y)| (x.clone(), y.clone())).unzip();
        Ok((stack_side(&xs)?, stack_side(&ys)?))
    }
}

pub struct DataLoader<D: Dataset> {
    pub dataset: Arc<D>,
    pub batch_size: usize,
    pub drop_last: bool,
    /// Number of prefetch worker threads (0 = inline, single-threaded).
    pub workers: usize,
    /// Outstanding batches per worker, including loading and reordered results.
    /// Zero uses one; the total window never exceeds the epoch's batch count.
    pub prefetch: usize,
    pub sampler: Box<dyn Sampler>,
    pub collate: Arc<dyn CollateFn>,
}

impl<D: Dataset + 'static> DataLoader<D> {
    pub fn new(dataset: Arc<D>, batch_size: usize) -> Self {
        if batch_size == 0 {
            panic!("DataLoader batch_size must be nonzero");
        }
        DataLoader {
            dataset,
            batch_size,
            drop_last: false,
            workers: 0,
            prefetch: 2,
            sampler: Box::new(SequentialSampler),
            collate: Arc::new(StackCollate),
        }
    }

    pub fn shuffle(mut self, seed: u64) -> Self {
        self.sampler = Box::new(RandomSampler { seed });
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }

    pub fn prefetch(mut self, prefetch: usize) -> Self {
        self.prefetch = prefetch;
        self
    }

    pub fn sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.sampler = sampler;
        self
    }

    pub fn collate_fn(mut self, collate: Arc<dyn CollateFn>) -> Self {
        self.collate = collate;
        self
    }

    pub fn len(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// One epoch of batches, in sampler order regardless of worker count.
    /// Sample-loading and collation errors/panics become ordered batch errors.
    pub fn iter(&self) -> Batches<'_, D> {
        let order = self.sampler.order(self.dataset.len());
        let mut slices: Vec<Vec<usize>> = order
            .chunks(self.batch_size)
            .map(<[usize]>::to_vec)
            .collect();
        if self.drop_last {
            if let Some(last) = slices.last() {
                if last.len() < self.batch_size {
                    slices.pop();
                }
            }
        }
        if self.workers == 0 || slices.len() <= 1 {
            return Batches {
                inline: Some(InlineIter {
                    ds: &*self.dataset,
                    collate: self.collate.clone(),
                    slices: slices.into(),
                }),
                threaded: None,
                _marker: std::marker::PhantomData,
            };
        }
        let nbatches = slices.len();
        let workers = self.workers.min(nbatches);
        let capacity = workers.saturating_mul(self.prefetch.max(1)).min(nbatches);
        let (tx, rx) =
            mpsc::sync_channel::<(usize, Result<(Tensor, Tensor)>)>(capacity);
        let slices = Arc::new(slices);
        let window = Arc::new(PrefetchWindow {
            state: Mutex::new(WindowState { next: 0, consumed: 0, cancelled: false }),
            changed: Condvar::new(), capacity, total: nbatches,
        });
        let mut threaded = ThreadedIter {
            rx: Some(rx), buffer: BTreeMap::new(), next_id: 0, nbatches,
            window: window.clone(), pending_error: None,
        };
        for _ in 0..workers {
            let tx = tx.clone();
            let slices = slices.clone();
            let window = window.clone();
            let ds = self.dataset.clone();
            let collate = self.collate.clone();
            let spawned = std::thread::Builder::new().spawn(move || loop {
                let Some(b) = window.claim() else { break };
                let batch = load_batch(&*ds, &*collate, &slices[b]);
                if tx.send((b, batch)).is_err() {
                    break;
                }
            });
            if let Err(error) = spawned {
                threaded.stop();
                threaded.pending_error = Some(Error::Unsupported { op: "DataLoader", msg: format!("cannot start prefetch worker: {error}") });
                break;
            }
        }
        drop(tx);
        Batches {
            inline: None,
            threaded: Some(threaded),
            _marker: std::marker::PhantomData,
        }
    }
}

fn load_batch(ds: &dyn Dataset, collate: &dyn CollateFn, indices: &[usize]) -> Result<(Tensor, Tensor)> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let items: Result<Vec<_>> = indices.iter().map(|&i| ds.get(i)).collect();
        items.and_then(|items| collate.collate(&items))
    })).unwrap_or_else(|panic| {
        let message = panic.downcast_ref::<String>().map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied()).unwrap_or("non-string panic");
        Err(Error::Unsupported { op: "DataLoader", msg: format!("dataset or collate panicked: {message}") })
    })
}

struct WindowState { next: usize, consumed: usize, cancelled: bool }

struct PrefetchWindow {
    state: Mutex<WindowState>,
    changed: Condvar,
    capacity: usize,
    total: usize,
}

impl PrefetchWindow {
    fn claim(&self) -> Option<usize> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while !state.cancelled && state.next < self.total && state.next - state.consumed >= self.capacity {
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        if state.cancelled || state.next >= self.total { return None; }
        let next = state.next;
        state.next += 1;
        Some(next)
    }

    fn cancel(&self) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).cancelled = true;
        self.changed.notify_all();
    }
}

struct InlineIter<'a> {
    ds: &'a dyn Dataset,
    collate: Arc<dyn CollateFn>,
    slices: std::collections::VecDeque<Vec<usize>>,
}

impl Iterator for InlineIter<'_> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = self.slices.pop_front()?;
        Some(load_batch(self.ds, &*self.collate, &slice))
    }
}

struct ThreadedIter {
    rx: Option<mpsc::Receiver<(usize, Result<(Tensor, Tensor)>)>>,
    buffer: BTreeMap<usize, Result<(Tensor, Tensor)>>,
    next_id: usize,
    nbatches: usize,
    window: Arc<PrefetchWindow>,
    pending_error: Option<Error>,
}

impl ThreadedIter {
    fn stop(&mut self) {
        self.next_id = self.nbatches;
        self.window.cancel();
        self.rx = None;
        self.buffer.clear();
    }
}

impl Drop for ThreadedIter {
    fn drop(&mut self) { self.stop(); }
}

impl Iterator for ThreadedIter {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(error) = self.pending_error.take() { return Some(Err(error)); }
        if self.next_id >= self.nbatches {
            return None;
        }
        while !self.buffer.contains_key(&self.next_id) {
            match self.rx.as_ref()?.recv() {
                Ok((b, batch)) => {
                    self.buffer.insert(b, batch);
                }
                Err(_) => {
                    let error = Error::Unsupported { op: "DataLoader", msg: format!("prefetch workers stopped before batch {}", self.next_id) };
                    self.stop();
                    return Some(Err(error));
                }
            }
        }
        let batch = self.buffer.remove(&self.next_id)?;
        self.next_id += 1;
        self.window.state.lock().unwrap_or_else(|e| e.into_inner()).consumed += 1;
        self.window.changed.notify_all();
        Some(batch)
    }
}

/// Epoch iterator over batches. Drop cancels new work and unblocks worker waits
/// and sends. Already-running dataset/collate calls finish independently.
pub struct Batches<'a, D: Dataset> {
    inline: Option<InlineIter<'a>>,
    threaded: Option<ThreadedIter>,
    _marker: std::marker::PhantomData<&'a D>,
}

impl<D: Dataset> Iterator for Batches<'_, D> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(it) = &mut self.inline {
            it.next()
        } else {
            self.threaded.as_mut()?.next()
        }
    }
}
