//! Required-GPU prerequisite gates for static model graph preparation.
use ferro_core::{Backend, UnaryKind};
use ferro_cuda::{CudaBackend, StaticPointwiseRun};
use std::sync::Arc;

#[test]
fn static_dag_retains_sources_and_replays_one_graph_without_allocations() {
    let Some(b) = backend() else { return };
    let b = Arc::new(b);
    let x: Arc<dyn ferro_core::dispatch::DeviceBuffer> = Arc::from(b.alloc_from_host(&[1.0, -2.0]).unwrap());
    let w: Arc<dyn ferro_core::dispatch::DeviceBuffer> = Arc::from(b.alloc_from_host(&[3.0, 4.0]).unwrap());
    let runs = vec![
        StaticPointwiseRun { inputs: vec![0], steps: vec![ferro_cuda::ChainStep::Unary(UnaryKind::Relu)] },
        StaticPointwiseRun { inputs: vec![0, 1], steps: vec![ferro_cuda::ChainStep::Binary { kind: ferro_core::BinaryKind::Mul, other: 1 }] },
        StaticPointwiseRun { inputs: vec![2, 3], steps: vec![ferro_cuda::ChainStep::Binary { kind: ferro_core::BinaryKind::Add, other: 1 }] },
    ];
    let mut graph = b.prepare_pointwise_graph(&runs, vec![x.clone(), w.clone()]).unwrap();
    assert_eq!(graph.kernel_nodes(), 3);
    let before = b.alloc_stats();
    let transfers = b.layer_norm_counts();
    let launches = b.pointwise_launch_counts();
    graph.replay().unwrap();
    assert_eq!(graph.replay_count(), 1);
    assert_eq!(b.alloc_stats().requests, before.requests);
    assert_eq!(b.layer_norm_counts(), transfers);
    assert_eq!(b.pointwise_launch_counts(), launches);
    let snapshot = graph.snapshot().unwrap();
    assert_eq!(b.copy_to_host(snapshot.as_ref()).unwrap(), vec![4.0, -8.0]);
    b.copy_into(x.as_ref(), &[-3.0, 2.0]).unwrap();
    b.copy_into(w.as_ref(), &[5.0, -1.0]).unwrap();
    drop(x);
    drop(w);
    let pressure: Vec<_> = (0..32).map(|_| b.alloc_from_host(&[77.0, 88.0]).unwrap()).collect();
    for _ in 0..4 {
        let before = b.alloc_stats();
        graph.replay().unwrap();
        assert_eq!(b.alloc_stats().requests, before.requests);
        assert_eq!(graph.copy_output_to_host().unwrap(), vec![-15.0, 0.0]);
        assert_eq!(b.copy_to_host(snapshot.as_ref()).unwrap(), vec![4.0, -8.0]);
    }
    drop(graph);
    drop(pressure);
    assert_eq!(b.copy_to_host(snapshot.as_ref()).unwrap(), vec![4.0, -8.0]);
}


#[test]
fn legacy_capture_cannot_disable_tracking_with_live_static_graph() {
    let Some(b) = backend() else { return };
    let b = Arc::new(b);
    let x: Arc<dyn ferro_core::dispatch::DeviceBuffer> = Arc::from(b.alloc_from_host(&[1.0]).unwrap());
    let runs = [StaticPointwiseRun { inputs: vec![0], steps: vec![ferro_cuda::ChainStep::Unary(UnaryKind::Relu)] }];
    let mut graph = b.prepare_pointwise_graph(&runs, vec![x]).unwrap();
    let start = b.begin_step_capture();
    if start.is_ok() { let _ = b.end_step_capture(); }
    assert!(start.is_err(), "legacy capture must not disable static graph tracking");
    let y = b.alloc_from_host(&[2.0]).unwrap();
    assert!(b.capture_chain(&[ferro_cuda::ChainStep::Unary(UnaryKind::Relu)], &[y.as_ref()]).is_err(), "legacy chain capture must not disable tracking");
    graph.replay().unwrap();
    assert_eq!(graph.copy_output_to_host().unwrap(), vec![1.0]);
}


#[test]
fn rejects_invalid_or_unsupported_dag_before_allocating() {
    let Some(b) = backend() else { return };
    let b = Arc::new(b);
    let x: Arc<dyn ferro_core::dispatch::DeviceBuffer> = Arc::from(b.alloc_from_host(&[1.0, 2.0]).unwrap());
    let invalid = vec![
        StaticPointwiseRun { inputs: vec![9], steps: vec![ferro_cuda::ChainStep::Unary(UnaryKind::Relu)] },
        StaticPointwiseRun { inputs: vec![0], steps: vec![ferro_cuda::ChainStep::Binary { kind: ferro_core::BinaryKind::Add, other: usize::MAX }] },
        StaticPointwiseRun { inputs: vec![0], steps: vec![] },
        StaticPointwiseRun { inputs: vec![0], steps: vec![ferro_cuda::ChainStep::BinaryBc { kind: ferro_core::BinaryKind::Add, other: 0, dims: vec![2], strides: vec![1] }] },
    ];
    for run in invalid {
        let before = b.alloc_stats();
        assert!(b.prepare_pointwise_graph(&[run], vec![x.clone()]).is_err());
        assert_eq!(b.alloc_stats().requests, before.requests);
    }
    let other = Arc::new(CudaBackend::new(0).unwrap());
    let run = StaticPointwiseRun { inputs: vec![0], steps: vec![ferro_cuda::ChainStep::Unary(UnaryKind::Relu)] };
    assert!(other.prepare_pointwise_graph(&[run], vec![x]).is_err());
}

fn backend() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required GPU unavailable: {e}"),
        Err(e) => { eprintln!("skipping GPU test: {e}"); None }
    }
}

#[test]
fn rejects_same_ordinal_foreign_stream_before_enqueue() {
    let Some(a) = backend() else { return };
    let b = CudaBackend::new(0).unwrap();
    let x = a.alloc_from_host(&[1.0, -2.0]).unwrap();
    let before = b.alloc_stats();
    let result = b.unary_dev(UnaryKind::Relu, x.as_ref());
    assert!(result.is_err(), "same device ordinal is not backend ownership");
    assert_eq!(b.alloc_stats().requests, before.requests);
    assert_eq!(a.copy_to_host(x.as_ref()).unwrap(), vec![1.0, -2.0]);
}
