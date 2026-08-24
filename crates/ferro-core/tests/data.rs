use ferro_core::data::{
    CollateFn, DataLoader, Dataset, RandomSampler, Sampler, StackCollate, TensorDataset,
};
use ferro_core::{Result, Tensor};
use std::sync::{Arc, Condvar, Mutex};

fn arange_dataset(n: usize) -> TensorDataset {
    let xs = Tensor::from_vec((0..n).map(|i| i as f32).collect(), &[n, 1]).unwrap();
    let ys = Tensor::from_vec((0..n).map(|i| (i * 3) as f32).collect(), &[n, 1]).unwrap();
    TensorDataset::new(xs, ys).unwrap()
}

fn batch_values(b: &Result<(Tensor, Tensor)>) -> Vec<f32> {
    let (x, _) = b.as_ref().unwrap();
    x.to_vec()
}

#[test]
fn batching_with_remainder() {
    let ds = Arc::new(arange_dataset(10));
    let loader = DataLoader::new(ds.clone(), 4);
    assert_eq!(loader.len(), 3);
    let sizes: Vec<usize> = loader.iter().map(|b| b.unwrap().0.shape()[0]).collect();
    assert_eq!(sizes, vec![4, 4, 2]);
    let vals: Vec<Vec<f32>> = loader.iter().map(|b| batch_values(&b)).collect();
    assert_eq!(vals[0], vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(vals[2], vec![8.0, 9.0]);

    let dropped = DataLoader::new(ds, 4).drop_last(true);
    assert_eq!(dropped.len(), 2);
    let sizes: Vec<usize> = dropped.iter().map(|b| b.unwrap().0.shape()[0]).collect();
    assert_eq!(sizes, vec![4, 4]);
}

#[test]
fn shuffle_determinism_given_seed() {
    let ds = Arc::new(arange_dataset(16));
    let run = || -> Vec<Vec<f32>> {
        DataLoader::new(ds.clone(), 4)
            .shuffle(42)
            .iter()
            .map(|b| batch_values(&b))
            .collect()
    };
    let a = run();
    let b = run();
    assert_eq!(a, b);
    let mut flat_a: Vec<f32> = a.concat();
    flat_a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(flat_a, (0..16).map(|i| i as f32).collect::<Vec<_>>());
    // Seed 42 must actually permute relative to sequential order.
    let seq: Vec<f32> = DataLoader::new(ds.clone(), 16)
        .iter()
        .map(|b| batch_values(&b))
        .collect::<Vec<_>>()
        .concat();
    assert_ne!(a.concat(), seq);
    // A different seed gives a different permutation for this size.
    let other: Vec<f32> = DataLoader::new(ds, 16)
        .shuffle(7)
        .iter()
        .map(|b| batch_values(&b))
        .collect::<Vec<_>>()
        .concat();
    assert_ne!(a.concat(), other);
}

#[test]
fn multiworker_matches_single_worker_ordering() {
    let ds = Arc::new(arange_dataset(23));
    let collect = |workers: usize, seed: u64| -> Vec<Vec<f32>> {
        DataLoader::new(ds.clone(), 3)
            .shuffle(seed)
            .workers(workers)
            .prefetch(1)
            .iter()
            .map(|b| batch_values(&b))
            .collect()
    };
    assert_eq!(collect(0, 123), collect(4, 123));
    assert_eq!(collect(1, 5), collect(8, 5));
}

/// Structural overlap test: every get() parks on a condvar gate until two
/// workers are simultaneously inside get(), which a helper thread observes
/// and then releases. Iteration can only finish if true concurrent access
/// happened; no timing is measured. A watchdog exits nonzero on deadlock.
#[test]
fn workers_overlap_prefetch() {
    #[derive(Default)]
    struct Gate {
        entered: usize,
        max_entered: usize,
        open: bool,
    }
    struct Shared {
        gate: Mutex<Gate>,
        cv: Condvar,
    }
    impl Shared {
        fn enter_and_wait(&self) {
            let mut g = self.gate.lock().unwrap();
            if g.open {
                return;
            }
            g.entered += 1;
            g.max_entered = g.max_entered.max(g.entered);
            self.cv.notify_all();
            while !g.open {
                g = self.cv.wait(g).unwrap();
            }
            g.entered -= 1;
            self.cv.notify_all();
        }
    }
    struct GatedDs {
        inner: TensorDataset,
        shared: Arc<Shared>,
    }
    impl Dataset for GatedDs {
        fn len(&self) -> usize {
            self.inner.len()
        }
        fn get(&self, idx: usize) -> Result<(Tensor, Tensor)> {
            self.shared.enter_and_wait();
            self.inner.get(idx)
        }
    }

    let shared = Arc::new(Shared {
        gate: Mutex::new(Gate::default()),
        cv: Condvar::new(),
    });
    let ds = Arc::new(GatedDs {
        inner: arange_dataset(6),
        shared: shared.clone(),
    });
    let loader = DataLoader::new(ds.clone(), 1).workers(2);

    let opener = {
        let shared = shared.clone();
        std::thread::spawn(move || {
            let mut g = shared.gate.lock().unwrap();
            while g.entered < 2 {
                g = shared.cv.wait(g).unwrap();
            }
            assert_eq!(g.max_entered, 2);
            g.open = true;
            shared.cv.notify_all();
        })
    };
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        if done_rx
            .recv_timeout(std::time::Duration::from_secs(20))
            .is_err()
        {
            eprintln!("worker overlap never happened - gate deadlocked");
            std::process::exit(101);
        }
    });

    let got: Vec<f32> = loader
        .iter()
        .map(|b| batch_values(&b))
        .collect::<Vec<_>>()
        .concat();
    let mut sorted = got;
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(sorted, (0..6).map(|i| i as f32).collect::<Vec<_>>());
    assert!(opener.join().is_ok());
    assert_eq!(shared.gate.lock().unwrap().max_entered, 2);
    done_tx.send(()).unwrap();
    watchdog.join().unwrap();
}

#[test]
fn custom_collate_fn() {
    struct DoubleCollate;
    impl CollateFn for DoubleCollate {
        fn collate(&self, items: &[(Tensor, Tensor)]) -> Result<(Tensor, Tensor)> {
            let stacked = StackCollate.collate(items)?;
            Ok((stacked.0.add(&stacked.0)?, stacked.1))
        }
    }
    let ds = Arc::new(arange_dataset(4));
    let out: Vec<f32> = DataLoader::new(ds, 4)
        .collate_fn(Arc::new(DoubleCollate))
        .iter()
        .map(|b| batch_values(&b))
        .collect::<Vec<_>>()
        .concat();
    assert_eq!(out, vec![0.0, 2.0, 4.0, 6.0]);
}

#[test]
fn random_sampler_is_deterministic_and_covers_all() {
    let s = RandomSampler { seed: 9 };
    assert_eq!(s.order(11), s.order(11));
    let mut o = s.order(11);
    o.sort_unstable();
    assert_eq!(o, (0..11).collect::<Vec<_>>());
}
