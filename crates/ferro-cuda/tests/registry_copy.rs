use ferro_core::{Backend, Device, UnaryKind};
use ferro_core::dispatch::DeviceBuffer;
use ferro_cuda::CudaBackend;

fn backend() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}");
            eprintln!("CUDA unavailable: {e}");
            None
        }
    }
}

#[test]
fn old_allocation_host_reads_survive_owner_drop_without_relaxing_compute() {
    let Some(old) = backend() else { return };
    let new = CudaBackend::new(0).unwrap();
    let values = [1.0, -2.5, 3.0];
    let x = old.alloc_from_host(&values).unwrap();
    assert_eq!(old.copy_to_host(&*x).unwrap(), values);
    // Two live owners must remain distinct for compute, mutation and D2D.
    for result in [new.unary_dev(UnaryKind::Neg, &*x).map(|_| ()),
        new.fill_inplace_dev(&*x, 9.0), new.copy_dev(&*x).map(|_| ())] {
        assert!(result.unwrap_err().to_string().contains("different CUDA backend stream/context"));
    }
    let ints = [i64::MIN, -(1_i64 << 54) - 1, (1_i64 << 54) + 1, i64::MAX];
    let ix = old.alloc_i64_from_host(&ints).unwrap();
    let empty = old.alloc_from_host(&[]).unwrap();
    let empty_i64 = old.alloc_i64_from_host(&[]).unwrap();
    assert!(new.copy_to_host(&*ix).is_err());
    assert!(new.copy_i64_to_host(&*x).is_err());
    drop(old);
    let before = new.layer_norm_counts();
    assert_eq!(new.copy_to_host(&*x).unwrap(), values);
    assert_eq!(new.copy_i64_to_host(&*ix).unwrap(), ints);
    assert!(new.copy_to_host(&*empty).unwrap().is_empty());
    assert!(new.copy_i64_to_host(&*empty_i64).unwrap().is_empty());
    assert_eq!(new.layer_norm_counts().2 - before.2, 2);
    let fresh = new.alloc_from_host(&values).unwrap();
    let neg = new.unary_dev(UnaryKind::Neg, &*fresh).unwrap();
    assert_eq!(new.copy_to_host(&*neg).unwrap(), [-1.0, 2.5, -3.0]);
}

#[test]
fn gather_rejects_index_buffers_from_another_owner_even_when_empty() {
    let Some(old) = backend() else { return };
    let new = CudaBackend::new(0).unwrap();
    let weights = new.alloc_from_host(&[1., 2., 3., 4.]).unwrap();
    let indices = old.alloc_i64_from_host(&[1, 0]).unwrap();
    let empty = old.alloc_i64_from_host(&[]).unwrap();
    drop(old);
    for index in [&*indices, &*empty] {
        let before = new.alloc_stats().requests;
        let result = new.gather_rows_dev(&*weights, index, 2, 2);
        let error = result.err().expect("foreign-owner gather must fail before allocation or launch");
        assert!(error.to_string().contains("different CUDA backend stream/context"));
        assert_eq!(new.alloc_stats().requests, before);
    }
    assert_eq!(new.copy_i64_to_host(&*indices).unwrap(), [1, 0]);
    let fresh = new.alloc_i64_from_host(&[1, 0]).unwrap();
    let out = new.gather_rows_dev(&*weights, &*fresh, 2, 2).unwrap();
    assert_eq!(new.copy_to_host(&*out).unwrap(), [3., 4., 1., 2.]);
    let empty = new.alloc_i64_from_host(&[]).unwrap();
    assert_eq!(new.gather_rows_dev(&*weights, &*empty, 2, 2).unwrap().len(), 0);
}

#[test]
fn tensor_gather_propagates_foreign_owner_errors_without_cpu_fallback() {
    use ferro_core::{Tensor, dispatch::register_backend};
    use std::sync::{Arc, Mutex};
    static SERIAL: Mutex<()> = Mutex::new(());
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(old) = backend() else { return };
    let old = Arc::new(old);
    let dev = Device::Cuda(0);
    register_backend(dev, old.clone());
    let old_weights = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap().to_device(dev).unwrap();
    let old_indices = Tensor::from_vec_i64(vec![1, 0], &[2]).unwrap().to_device(dev).unwrap();
    let old_empty = Tensor::from_vec_i64(vec![], &[0]).unwrap().to_device(dev).unwrap();
    let new = Arc::new(CudaBackend::new(0).unwrap());
    register_backend(dev, new.clone());
    drop(old);
    let weights = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap().to_device(dev).unwrap();
    let indices = Tensor::from_vec_i64(vec![1, 0], &[2]).unwrap().to_device(dev).unwrap();
    for (w, idx) in [(&weights, &old_indices), (&weights, &old_empty), (&old_weights, &indices)] {
        let requests = new.alloc_stats().requests;
        let downloads = new.layer_norm_counts().2;
        let error = w.index_select_t(0, idx).err().expect("foreign-owner tensor gather must fail");
        assert!(error.to_string().contains("different CUDA backend stream/context"));
        assert_eq!(new.alloc_stats().requests, requests);
        assert_eq!(new.layer_norm_counts().2, downloads);
    }
    assert_eq!(old_indices.to_vec_i64(), [1, 0]);
    let y = weights.index_select_t(0, &indices).unwrap();
    assert_eq!(y.device(), dev);
    assert_eq!(y.to_vec(), [3., 4., 1., 2.]);
}

struct Foreign(Device);
impl DeviceBuffer for Foreign {
    fn device(&self) -> Device { self.0 }
    fn len(&self) -> usize { 0 }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

#[test]
fn host_reads_reject_foreign_buffer_types_even_when_empty() {
    let Some(b) = backend() else { return };
    for device in [Device::Cpu, Device::Cuda(0), Device::Cuda(1)] {
        assert!(b.copy_to_host(&Foreign(device)).is_err());
        assert!(b.copy_i64_to_host(&Foreign(device)).is_err());
    }
}
