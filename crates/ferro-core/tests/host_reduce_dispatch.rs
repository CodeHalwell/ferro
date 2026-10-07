//! Host sum/mean/sum_dim/softmax/log_softmax route through the Cpu backend's
//! seams, so an optimized backend (ferro-fastcpu) can take them. This binary
//! has exactly one test because it replaces the process-global Cpu backend.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ferro_core::dispatch::{register_backend, Backend, BinaryKind, UnaryKind};
use ferro_core::{CpuBackend, Device, Tensor};

static CALLS: [AtomicUsize; 4] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];

/// CpuBackend with counted, sentinel-offset host reductions.
struct Counting;

impl Backend for Counting {
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        CpuBackend.unary(kind, x)
    }
    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
        CpuBackend.binary(kind, a, b)
    }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        CpuBackend.matmul(a, b, m, k, n)
    }
    fn sum(&self, x: &[f32]) -> f32 {
        CALLS[0].fetch_add(1, Ordering::SeqCst);
        CpuBackend.sum(x) + 1000.0
    }
    fn sum_dim(&self, x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
        CALLS[1].fetch_add(1, Ordering::SeqCst);
        CpuBackend.sum_dim(x, shape, dim).into_iter().map(|v| v + 1000.0).collect()
    }
    fn softmax(&self, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        CALLS[2].fetch_add(1, Ordering::SeqCst);
        CpuBackend.softmax(x, rows, cols).into_iter().map(|v| v + 1000.0).collect()
    }
    fn log_softmax(&self, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        CALLS[3].fetch_add(1, Ordering::SeqCst);
        CpuBackend.log_softmax(x, rows, cols).into_iter().map(|v| v + 1000.0).collect()
    }
}

fn calls() -> [usize; 4] {
    CALLS.each_ref().map(|c| c.load(Ordering::SeqCst))
}

#[test]
fn host_reductions_route_through_cpu_backend() {
    let x = Tensor::from_vec((0..24).map(|i| (i as f32 * 0.37).sin()).collect(), &[2, 3, 4]).unwrap();
    let v = x.to_vec();
    let sum = CpuBackend.sum(&v);
    let xt = x.transpose(0, 2).unwrap();
    let sum_t = CpuBackend.sum(&xt.to_vec());
    let rows = CpuBackend.softmax(&v, 6, 4);
    let log_rows = CpuBackend.log_softmax(&v, 6, 4);
    let mid = x.softmax(1).unwrap().to_vec();
    let log_mid = x.log_softmax(1).unwrap().to_vec();
    register_backend(Device::Cpu, Arc::new(Counting));

    assert_eq!(x.sum().to_vec(), vec![sum + 1000.0]);
    assert_eq!(x.mean().to_vec(), vec![(sum + 1000.0) / 24.0]);
    // A transposed view takes the pooled-copy path, same seam.
    assert_eq!(xt.sum().to_vec(), vec![sum_t + 1000.0]);
    assert_eq!(calls(), [3, 0, 0, 0]);

    let want: Vec<f32> = CpuBackend.sum_dim(&v, &[2, 3, 4], 1).into_iter().map(|s| s + 1000.0).collect();
    assert_eq!(x.sum_dim(1, false).unwrap().to_vec(), want);
    assert_eq!(calls(), [3, 1, 0, 0]);

    // Only the last dim maps onto the rows x cols seam; other dims keep the
    // in-core strided reference and never call the backend.
    assert_eq!(x.softmax(2).unwrap().to_vec(), rows.iter().map(|v| v + 1000.0).collect::<Vec<_>>());
    assert_eq!(x.log_softmax(2).unwrap().to_vec(), log_rows.iter().map(|v| v + 1000.0).collect::<Vec<_>>());
    assert_eq!(x.softmax(1).unwrap().to_vec(), mid);
    assert_eq!(x.log_softmax(1).unwrap().to_vec(), log_mid);
    assert_eq!(calls(), [3, 1, 1, 1]);
    register_backend(Device::Cpu, Arc::new(CpuBackend));
}
