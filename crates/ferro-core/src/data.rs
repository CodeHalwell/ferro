//! Data loading: datasets, samplers, collate functions and a DataLoader with
//! std::thread-based prefetch workers. ferro-core stays zero-dependency, so
//! worker parallelism uses plain threads plus mpsc channels.

use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::tensor::Tensor;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

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

/// Default collate: stacks f32 items along a new leading dimension.
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
    /// Upper bound on batches buffered ahead of the consumer per cycle.
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
        let (tx, rx) =
            mpsc::sync_channel::<(usize, Result<(Tensor, Tensor)>)>(workers * self.prefetch.max(1));
        let slices = Arc::new(slices);
        let next_batch = Arc::new(AtomicUsize::new(0));
        for _ in 0..workers {
            let tx = tx.clone();
            let slices = slices.clone();
            let next_batch = next_batch.clone();
            let ds = self.dataset.clone();
            let collate = self.collate.clone();
            std::thread::spawn(move || loop {
                let b = next_batch.fetch_add(1, Ordering::SeqCst);
                if b >= slices.len() {
                    break;
                }
                let items: Result<Vec<(Tensor, Tensor)>> =
                    slices[b].iter().map(|&i| ds.get(i)).collect();
                let batch = items.and_then(|items| collate.collate(&items));
                if tx.send((b, batch)).is_err() {
                    break;
                }
            });
        }
        drop(tx);
        Batches {
            inline: None,
            threaded: Some(ThreadedIter {
                rx,
                buffer: BTreeMap::new(),
                next_id: 0,
                nbatches,
            }),
            _marker: std::marker::PhantomData,
        }
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
        let items: Result<Vec<(Tensor, Tensor)>> = slice.iter().map(|&i| self.ds.get(i)).collect();
        Some(items.and_then(|items| self.collate.collate(&items)))
    }
}

struct ThreadedIter {
    rx: mpsc::Receiver<(usize, Result<(Tensor, Tensor)>)>,
    buffer: BTreeMap<usize, Result<(Tensor, Tensor)>>,
    next_id: usize,
    nbatches: usize,
}

impl Iterator for ThreadedIter {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_id >= self.nbatches {
            return None;
        }
        while !self.buffer.contains_key(&self.next_id) {
            match self.rx.recv() {
                Ok((b, batch)) => {
                    self.buffer.insert(b, batch);
                }
                Err(_) => return None,
            }
        }
        let batch = self.buffer.remove(&self.next_id)?;
        self.next_id += 1;
        Some(batch)
    }
}

/// Epoch iterator over batches. Dropping it stops the workers.
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
