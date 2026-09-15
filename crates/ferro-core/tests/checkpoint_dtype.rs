use ferro_core::{checkpoint::Checkpoint, device::Device, dtype::DType, nn::Module, params::Param, tensor::Tensor};

#[test]
fn named_snapshot_returns_device_copy_errors() {
    use ferro_core::dispatch::{Backend, BinaryKind, UnaryKind, register_backend};
    use ferro_core::dispatch::DeviceBuffer;
    use std::sync::Arc;
    struct Buffer;
    impl DeviceBuffer for Buffer {
        fn device(&self) -> Device { Device::Cuda(79) }
        fn len(&self) -> usize { 4 }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }
    struct Failure;
    impl Backend for Failure {
        fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { unreachable!() }
        fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { unreachable!() }
        fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { unreachable!() }
        fn alloc_from_host(&self, _: &[f32]) -> ferro_core::Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buffer)) }
        fn alloc_i64_from_host(&self, _: &[i64]) -> ferro_core::Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buffer)) }
        fn copy_to_host(&self, _: &dyn DeviceBuffer) -> ferro_core::Result<Vec<f32>> {
            Err(ferro_core::Error::Io { op: "copy_to_host", msg: "injected copy failure".into() })
        }
        fn copy_i64_to_host(&self, _: &dyn DeviceBuffer) -> ferro_core::Result<Vec<i64>> {
            Err(ferro_core::Error::Io { op: "copy_i64_to_host", msg: "injected copy failure".into() })
        }
        fn copy_dev(&self, _: &dyn DeviceBuffer) -> ferro_core::Result<Box<dyn DeviceBuffer>> {
            Err(ferro_core::Error::Io { op: "copy_dev", msg: "injected copy failure".into() })
        }
    }
    register_backend(Device::Cuda(79), Arc::new(Failure));
    for (dtype, transpose, op) in [(DType::F32, false, "copy_dev"), (DType::F32, true, "copy_to_host"),
        (DType::I64, false, "copy_i64_to_host"), (DType::I64, true, "copy_i64_to_host")] {
        let t = Tensor::ones(&[2, 2]).to_dtype(dtype).to_device(Device::Cuda(79)).unwrap();
        let t = if transpose { t.transpose(0, 1).unwrap() } else { t };
        let error = Checkpoint::new(1).with_named_state(&[("value".into(), t)]).err();
        assert_eq!(error, Some(ferro_core::Error::Io { op, msg: "injected copy failure".into() }));
    }
}

fn samples() -> Vec<Tensor> {
    vec![
        Tensor::from_vec(vec![f32::from_bits(0x7fc01234), -0.0, 1.0, -2.0], &[2, 2]).unwrap(),
        Tensor::from_vec_f64(vec![f64::from_bits(0x7ff8000000001234), -0.0, 1.0000000000000002, -1e200], &[2, 2]).unwrap(),
        Tensor::from_vec_i64(vec![16_777_217, i64::MAX, i64::MIN, -16_777_217], &[2, 2]).unwrap(),
        Tensor::from_vec_f16_bits(vec![0x7d01, 0x8000, 0x0001, 0xfc00], &[2, 2]).unwrap(),
        Tensor::from_vec_bf16_bits(vec![0x7f81, 0x8000, 0x0001, 0xff80], &[2, 2]).unwrap(),
    ]
}

fn bits(t: &Tensor) -> Vec<u64> {
    match t.dtype() {
        DType::F32 => t.to_vec().iter().map(|v| v.to_bits() as u64).collect(),
        DType::F64 => t.to_vec_f64().iter().map(|v| v.to_bits()).collect(),
        DType::I64 => t.to_vec_i64().iter().map(|v| *v as u64).collect(),
        DType::F16 => t.to_vec_f16_bits().unwrap().iter().map(|v| *v as u64).collect(),
        DType::BF16 => t.to_vec_bf16_bits().unwrap().iter().map(|v| *v as u64).collect(),
    }
}

fn exact(expected: &Tensor, actual: &Tensor) {
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.device(), expected.device());
    assert_eq!(bits(actual), bits(expected));
    assert_ne!(actual._storage_ptr(), expected._storage_ptr());
    assert!(!actual.requires_grad());
}

struct Buffers(Vec<(String, Tensor)>);
impl Module for Buffers {
    fn forward(&self, x: &Tensor) -> ferro_core::Result<Tensor> { Ok(x.clone()) }
    fn named_parameters(&self) -> Vec<(String, Param)> { vec![] }
    fn named_buffers(&self) -> Vec<(String, Tensor)> { self.0.clone() }
}

fn typed_roundtrip(index: usize) {
    let dir = std::env::temp_dir().join(format!("ferro_checkpoint_dtype_{}_{index}", std::process::id()));
    for original in [samples().remove(index)] {
        for t in [original.clone(), original.transpose(0, 1).unwrap(), Tensor::zeros(&[0, 2]).to_dtype(original.dtype())] {
            let cp = Checkpoint::new(17).with_tensor("value", t.clone());
            exact(&t, &cp.tensors[0].1);
            // Clone also covers tensors supplied directly by file loading/public fields.
            let mut direct = Checkpoint::new(17);
            direct.tensors.push(("value".into(), t.clone()));
            exact(&t, &direct.clone().tensors[0].1);
            let named = Checkpoint::new(17).with_named_state(&[("value".into(), t.clone())]).unwrap();
            exact(&t, &named.tensors[0].1);
            let module = Checkpoint::from_module(17, &Buffers(vec![("value".into(), t.clone())]));
            exact(&t, &module.tensors[0].1);
            for snapshot in [cp.clone(), named, module] {
                snapshot.save_to_dir(&dir).unwrap();
                let loaded = Checkpoint::load_from_dir(&dir).unwrap();
                exact(&t, &loaded.tensors[0].1);
                exact(&t, &loaded.clone().tensors[0].1);
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn f32_checkpoint_roundtrip() { typed_roundtrip(0); }
#[test]
fn f64_checkpoint_roundtrip() { typed_roundtrip(1); }
#[test]
fn i64_checkpoint_roundtrip() { typed_roundtrip(2); }
#[test]
fn f16_checkpoint_roundtrip() { typed_roundtrip(3); }
#[test]
fn bf16_checkpoint_roundtrip() { typed_roundtrip(4); }

#[test]
fn checkpoint_snapshots_do_not_alias_live_values_or_each_other() {
    let source = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let cp = Checkpoint::new(1).with_tensor("value", source.transpose(0, 1).unwrap());
    let cloned = cp.clone();
    source.fill_(9.0).unwrap();
    assert_eq!(cp.tensors[0].1.to_vec(), [1.0, 3.0, 2.0, 4.0]);
    cp.tensors[0].1.fill_(8.0).unwrap();
    assert_eq!(cloned.tensors[0].1.to_vec(), [1.0, 3.0, 2.0, 4.0]);
    assert_eq!(cloned.tensors[0].1.device(), Device::Cpu);
}

#[test]
fn typed_io_does_not_enable_typed_transactional_restore() {
    let t = Tensor::from_vec_i64(vec![16_777_217], &[1]).unwrap();
    let cp = Checkpoint::new(1).with_tensor("value", t.clone());
    assert!(cp.load_named_state_into(&[("value".into(), t.clone())]).is_err());
    assert_eq!(t.to_vec_i64(), [16_777_217]);
}
