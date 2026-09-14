use ferro_core::{checkpoint::Checkpoint, device::Device, dispatch::register_backend, dtype::DType, tensor::Tensor};
use ferro_cuda::CudaBackend;
use std::sync::Arc;

#[test]
fn checkpoint_device_copies_preserve_dtype_placement_and_ownership() {
    let backend = match CudaBackend::new(0) {
        Ok(b) => Arc::new(b),
        Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required CUDA unavailable: {e}"),
        Err(e) => { eprintln!("skipping CUDA: {e}"); return; }
    };
    let device = Device::Cuda(0);
    register_backend(device, backend);
    let dir = std::env::temp_dir().join(format!("ferro_checkpoint_dtype_gpu_{}", std::process::id()));
    for cpu in [Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
        Tensor::from_vec_i64(vec![16_777_217, i64::MAX, i64::MIN, -16_777_217], &[2, 2]).unwrap()] {
        for transpose in [false, true] {
            let source = cpu.to_device(device).unwrap();
            let view = if transpose { source.transpose(0, 1).unwrap() } else { source.clone() };
            let expected = view.to_vec_i64();
            let cp = Checkpoint::new(7).with_tensor("value", view);
            let cloned = cp.clone();
            for snapshot in [&cp, &cloned] {
                let value = &snapshot.tensors[0].1;
                assert_eq!(value.device(), device);
                assert_eq!(value.dtype(), cpu.dtype());
                assert_eq!(value.to_vec_i64(), expected);
                assert_ne!(value._storage_ptr(), source._storage_ptr());
                snapshot.save_to_dir(&dir).unwrap();
                let loaded = Checkpoint::load_from_dir(&dir).unwrap();
                assert_eq!(loaded.tensors[0].1.dtype(), cpu.dtype());
                assert_eq!(loaded.tensors[0].1.to_vec_i64(), expected);
            }
            if cpu.dtype() == DType::F32 {
                source.fill_(9.0).unwrap();
                assert_eq!(cp.tensors[0].1.to_vec_i64(), expected);
                cp.tensors[0].1.fill_(8.0).unwrap();
                assert_eq!(cloned.tensors[0].1.to_vec_i64(), expected);
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
