//! Deterministic AB-BA schedule against the ordered exclusive replay sweep.
use super::*;
use crate::dispatch::{register_backend, AdamWStep, Backend};
use std::any::Any;
use std::cell::RefCell;
use std::process::Command;
use std::sync::{mpsc, Weak};
use std::time::{Duration, Instant};

const DEV: Device = Device::Cuda(31);
thread_local! {
    static WATCH: RefCell<Vec<Weak<StorageCell>>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn before_read(cell: &StorageCell) {
    WATCH.with(|watch| {
        let cells: Vec<_> = watch.borrow().iter().filter_map(Weak::upgrade).collect();
        let address = cell as *const StorageCell;
        for higher in &cells {
            if Arc::as_ptr(higher) > address && higher.data.try_write().is_err() {
                let lower = cells.iter().find(|c| Arc::as_ptr(c) == address).unwrap().clone();
                let higher = higher.clone();
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || {
                    // Exactly the prepared-replay sweep: low exclusive then high exclusive.
                    let _low = lower.write();
                    tx.send(()).unwrap();
                    let _high = higher.write();
                });
                rx.recv_timeout(Duration::from_secs(1)).unwrap();
                eprintln!("REPRO: copy/mutation holds high read; replay holds low write and waits high");
                // The caller now waits for low: a genuine cycle, killed by the parent.
                break;
            }
        }
    });
}

struct Buf(Vec<f32>);
impl DeviceBuffer for Buf {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}
struct Fake;
impl Backend for Fake {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { unreachable!() }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { unreachable!() }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { unreachable!() }
    fn copy_into_dev(&self, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer) -> Result<()> { Ok(()) }
    fn binary_inplace_dev(&self, _: BinaryKind, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer) -> Result<()> { Ok(()) }
    fn axpy_inplace_dev(&self, _: f32, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer) -> Result<()> { Ok(()) }
    fn sgd_step_dev(&self, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer, _: f32, _: f32, _: bool) -> Result<()> { Ok(()) }
    fn adamw_step_dev(&self, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer, _: AdamWStep) -> Result<()> { Ok(()) }
    fn adamw_step_capturable_dev(&self, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer, _: &dyn DeviceBuffer, _: f32, _: f32, _: f32, _: f32, _: f32) -> Result<()> { Ok(()) }
}

#[test]
fn device_mutation_replay_lock_order() {
    if let Ok(case) = std::env::var("FERRO_LOCK_CHILD") {
        register_backend(DEV, Arc::new(Fake));
        let mut ts: Vec<_> = (0..5).map(|_| device_leaf(Box::new(Buf(vec![1.; 3])), &[3], DEV)).collect();
        ts.sort_by_key(|t| Arc::as_ptr(&t.0.storage));
        WATCH.with(|w| *w.borrow_mut() = ts.iter().map(|t| Arc::downgrade(&t.0.storage)).collect());
        let [a,b,c,d,e] = <&[Tensor;5]>::try_from(ts.as_slice()).unwrap();
        match case.as_str() {
            "copy-desc" => e.copy_from(a).unwrap(),
            "copy-asc" => a.copy_from(e).unwrap(),
            "binary-desc" => e.add_(a).unwrap(),
            "binary-asc" => a.add_(e).unwrap(),
            "axpy" => crate::inplace::raw_axpy_("test", 1., e, a).unwrap(),
            "sgd" => crate::inplace::raw_sgd_step_(e,d,a,0.1,0.9,false).unwrap(),
            "adamw" => crate::inplace::raw_adamw_step_(e,d,c,a,AdamWStep { lr:0.1,beta1:0.9,beta2:0.99,eps:1e-8,weight_decay:0.,bc1:0.1,bc2:0.01 }).unwrap(),
            "capturable" => crate::inplace::raw_adamw_step_capturable_(e,d,c,b,a,0.1,0.9,0.99,1e-8,0.).unwrap(),
            "self" => { e.copy_from(e).unwrap(); e.add_(e).unwrap(); },
            _ => unreachable!(),
        }
        WATCH.with(|w| w.borrow_mut().clear());
        return;
    }
    let mut failures = Vec::new();
    for case in ["copy-desc","copy-asc","binary-desc","binary-asc","axpy","sgd","adamw","capturable","self"] {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tensor::concurrent_copy_tests::device_mutation_replay_lock_order", "--nocapture"])
            .env("FERRO_LOCK_CHILD", case).spawn().unwrap();
        let start = Instant::now();
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                if !status.success() { failures.push(format!("{case}: {status}")); }
                break;
            }
            if start.elapsed() > Duration::from_secs(3) {
                child.kill().unwrap(); child.wait().unwrap();
                failures.push(format!("{case}: timed out in ordered replay/mutation lock cycle"));
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
