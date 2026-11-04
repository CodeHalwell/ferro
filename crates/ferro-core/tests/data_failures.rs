use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::Duration;

use ferro_core::data::{CollateFn, DataLoader, Dataset, StackCollate, TensorDataset};
use ferro_core::{Error, Result, Tensor};

struct FailingDataset { panic: bool, fail: bool }
impl Dataset for FailingDataset {
    fn len(&self) -> usize { 6 }
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        if self.fail && index == 2 {
            if self.panic { panic!("dataset failure"); }
            return Err(Error::Unsupported { op: "dataset", msg: "item failed".into() });
        }
        Ok((Tensor::scalar(index as f32), Tensor::scalar(index as f32)))
    }
}

struct FailingCollate { panic: bool }
impl CollateFn for FailingCollate {
    fn collate(&self, items: &[(Tensor, Tensor)]) -> Result<(Tensor, Tensor)> {
        if items[0].0.item() == 2.0 {
            if self.panic { panic!("collate failure"); }
            return Err(Error::Unsupported { op: "collate", msg: "batch failed".into() });
        }
        StackCollate.collate(items)
    }
}

#[test]
fn dataset_and_collate_failures_keep_order_and_remaining_batches() {
    for workers in [0, 1, 3] {
        for panic in [false, true] {
            for dataset_failure in [false, true] {
                let mut loader = DataLoader::new(Arc::new(FailingDataset { panic, fail: dataset_failure }), 1).workers(workers).prefetch(1);
                if !dataset_failure { loader = loader.collate_fn(Arc::new(FailingCollate { panic })); }
                let batches: Vec<_> = loader.iter().collect();
                assert_eq!(batches.len(), 6);
                for (index, batch) in batches.into_iter().enumerate() {
                    if index == 2 {
                        let error = batch.err().expect("failed batch must remain visible");
                        if panic { assert!(error.to_string().contains("panicked")); }
                        else { assert!(error.to_string().contains("failed")); }
                    } else {
                        assert_eq!(batch.unwrap().0.item(), index as f32);
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct GateState { entered: usize, released: bool }
struct Gate { state: Mutex<GateState>, changed: Condvar }

struct GatedDataset { gate: Arc<Gate>, all: bool, dropped: Option<mpsc::Sender<()>> }
impl Dataset for GatedDataset {
    fn len(&self) -> usize { 32 }
    fn get(&self, index: usize) -> Result<(Tensor, Tensor)> {
        let mut state = self.gate.state.lock().unwrap_or_else(|e| e.into_inner());
        state.entered += 1;
        self.gate.changed.notify_all();
        while (self.all || index == 0) && !state.released {
            state = self.gate.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        Ok((Tensor::scalar(index as f32), Tensor::scalar(index as f32)))
    }
}
impl Drop for GatedDataset {
    fn drop(&mut self) { if let Some(tx) = &self.dropped { let _ = tx.send(()); } }
}

#[test]
fn slow_first_batch_cannot_expand_the_reorder_buffer_past_prefetch() {
    let capacity = 2;
    let gate = Arc::new(Gate { state: Mutex::new(GateState::default()), changed: Condvar::new() });
    let ds = Arc::new(GatedDataset { gate: gate.clone(), all: false, dropped: None });
    let (done, receive) = mpsc::channel();
    let consumer = std::thread::spawn(move || {
        let loader = DataLoader::new(ds, 1).workers(2).prefetch(1);
        let values: Vec<_> = loader.iter().map(|batch| batch.unwrap().0.item()).collect();
        done.send(values).unwrap();
    });
    let state = gate.state.lock().unwrap_or_else(|e| e.into_inner());
    let (state, _) = gate.changed.wait_timeout_while(state, Duration::from_secs(5), |s| s.entered < capacity).unwrap();
    // Keep the head blocked while the other worker attempts more work. The
    // timeout bounds this observation; the assertion measures actual calls.
    let (mut state, _) = gate.changed.wait_timeout_while(state, Duration::from_millis(250), |s| s.entered <= capacity).unwrap();
    let entered = state.entered;
    state.released = true;
    gate.changed.notify_all();
    drop(state);
    let values = receive.recv_timeout(Duration::from_secs(5)).expect("ordered epoch must complete after releasing the head");
    consumer.join().unwrap();
    assert_eq!(entered, capacity, "later batches bypassed the outstanding-work bound");
    assert_eq!(values, (0..32).map(|i| i as f32).collect::<Vec<_>>());
}

#[test]
fn dropping_prefetch_cancels_waiters_without_joining_running_callbacks() {
    for running_callbacks in [false, true] {
        let gate = Arc::new(Gate { state: Mutex::new(GateState { entered: 0, released: !running_callbacks }), changed: Condvar::new() });
        let (destroyed, destruction) = mpsc::channel();
        let ds = Arc::new(GatedDataset { gate: gate.clone(), all: true, dropped: Some(destroyed) });
        let (drop_now, may_drop) = mpsc::channel();
        let (dropped, drop_finished) = mpsc::channel();
        let owner = std::thread::spawn(move || {
            let loader = DataLoader::new(ds, 1).workers(2).prefetch(1);
            let batches = loader.iter();
            may_drop.recv().unwrap();
            drop(batches);
            dropped.send(()).unwrap();
        });
        let state = gate.state.lock().unwrap_or_else(|e| e.into_inner());
        let (state, _) = gate.changed.wait_timeout_while(state, Duration::from_secs(5), |s| s.entered < 2).unwrap();
        let entered = state.entered;
        drop(state);
        drop_now.send(()).unwrap();
        let finished = drop_finished.recv_timeout(Duration::from_secs(5));
        let mut state = gate.state.lock().unwrap_or_else(|e| e.into_inner());
        state.released = true;
        gate.changed.notify_all();
        drop(state);
        finished.expect("dropping the iterator must not wait for a blocked callback");
        owner.join().unwrap();
        destruction.recv_timeout(Duration::from_secs(5)).expect("workers must release the dataset after callbacks finish");
        assert_eq!(entered, 2);
        assert_eq!(gate.state.lock().unwrap().entered, 2, "drop must cancel unclaimed work");
    }
}

#[test]
fn extreme_prefetch_and_worker_settings_are_bounded_by_epoch_size() {
    let dataset = Arc::new(TensorDataset::new(Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3, 1]).unwrap(),
        Tensor::from_vec_i64(vec![i64::MAX, i64::MIN, 16_777_217], &[3]).unwrap()).unwrap());
    for prefetch in [0, usize::MAX] {
        let loader = DataLoader::new(dataset.clone(), 2).workers(usize::MAX).prefetch(prefetch);
        let batches: Vec<_> = loader.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches.iter().map(|(x, _)| x.shape()[0]).collect::<Vec<_>>(), vec![2, 1]);
        assert_eq!(batches.iter().flat_map(|(_, y)| y.to_vec_i64()).collect::<Vec<_>>(), vec![i64::MAX, i64::MIN, 16_777_217]);
    }
    let empty = Arc::new(TensorDataset::new(Tensor::zeros(&[0, 1]), Tensor::from_vec_i64(vec![], &[0]).unwrap()).unwrap());
    assert_eq!(DataLoader::new(empty, 2).workers(usize::MAX).prefetch(usize::MAX).iter().count(), 0);
}
