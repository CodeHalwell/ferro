use std::any::Any;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};

use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use ferro_core::dispatch::{Backend, BinaryKind, DeviceBuffer, UnaryKind};
use ferro_core::{DType, Device, Error, Result, Tensor, register_backend};

const DEV: Device = Device::Cuda(246);
static LOCK: Mutex<()> = Mutex::new(());
static CALLS: AtomicUsize = AtomicUsize::new(0);

struct Buffer(usize);
impl DeviceBuffer for Buffer {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0 }
    fn as_any(&self) -> &dyn Any { self }
}

struct Counter;
impl Backend for Counter {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected compute") }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected compute") }
    fn alloc_from_host(&self, values: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Buffer(values.len())))
    }
    fn alloc_i64_from_host(&self, values: &[i64]) -> Result<Box<dyn DeviceBuffer>> {
        CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Buffer(values.len())))
    }
    fn fill_dev(&self, _: f32, _: usize) -> Result<Box<dyn DeviceBuffer>> {
        CALLS.fetch_add(1, Ordering::SeqCst);
        Err(Error::Unsupported { op: "test_fill", msg: "injected allocation failure".into() })
    }
    fn copy_to_host(&self, _: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        CALLS.fetch_add(1, Ordering::SeqCst);
        panic!("unexpected download")
    }
}

#[test]
fn invalid_functional_operands_reject_before_device_work() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(Counter));
    let device = Tensor::scalar(2.0).to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let integer = Tensor::from_vec_i64(vec![2], &[]).unwrap().to_device(DEV).unwrap();
    let double = Tensor::from_vec_f64(vec![2.0], &[]).unwrap();
    let host = Tensor::scalar(2.0).requires_grad_(true).unwrap();
    host.mul(&host).unwrap().backward();
    for create_graph in [false, true] {
        for enabled in [false, true] {
            with_grad_enabled(enabled, || {
                for (output, input, expected) in [
                    (&integer, &device, Error::DtypeMismatch { op: "vjp_wrt", expected: DType::F32, got: DType::I64 }),
                    (&device, &integer, Error::DtypeMismatch { op: "vjp_wrt", expected: DType::F32, got: DType::I64 }),
                    (&device, &double, if create_graph { Error::DtypeMismatch { op: "vjp_wrt", expected: DType::F32, got: DType::F64 } } else { Error::DeviceMismatch { op: "vjp_wrt", lhs: DEV, rhs: Device::Cpu } }),
                    (&device, &host, Error::DeviceMismatch { op: "vjp_wrt", lhs: DEV, rhs: Device::Cpu }),
                ] {
                    CALLS.store(0, Ordering::SeqCst);
                    assert_eq!(output.grad_wrt(&[input], create_graph).err(), Some(expected.clone()));
                    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
                    assert_eq!(output.vjp_wrt(&[input], &device, create_graph).err(), Some(expected));
                    assert_eq!(CALLS.load(Ordering::SeqCst), 0);
                    assert_eq!(is_grad_enabled(), enabled);
                    assert!(device.grad().is_none());
                    assert_eq!(host.grad().unwrap().item(), 4.0);
                }
                assert!(matches!(device.vjp_wrt(&[&device], &host, create_graph), Err(Error::DeviceMismatch { .. })));
                assert!(matches!(device.vjp_wrt(&[&device], &integer, create_graph), Err(Error::DtypeMismatch { .. })));
                assert_eq!(CALLS.load(Ordering::SeqCst), 0);
                assert_eq!(is_grad_enabled(), enabled);
                assert!(matches!(device.grad_wrt(&[&device], create_graph), Err(Error::Unsupported { op: "test_fill", .. })));
                assert_eq!(CALLS.load(Ordering::SeqCst), 1, "valid metadata reaches the backend");
                assert_eq!(is_grad_enabled(), enabled);
            });
        }
    }
}
