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
