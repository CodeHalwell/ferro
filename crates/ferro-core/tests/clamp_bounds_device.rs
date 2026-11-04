// Counting fake device storage; no hardware, driver or CUDA execution claim.
use std::any::Any;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use ferro_core::{Device, Result, Tensor};
use ferro_core::dispatch::{Backend, BinaryKind, CpuBackend, DeviceBuffer, UnaryKind, register_backend};

const DEV: Device = Device::Cuda(220);
static SERIAL: Mutex<()> = Mutex::new(());
struct Buffer(Vec<f32>);
impl DeviceBuffer for Buffer {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}
#[derive(Default)]
struct Counting { uploads: AtomicUsize, downloads: AtomicUsize, kernels: AtomicUsize }
impl Counting {
    fn counts(&self) -> [usize; 3] { [&self.uploads, &self.downloads, &self.kernels].map(|count| count.load(Ordering::SeqCst)) }
}
impl Backend for Counting {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected host backend call") }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected host backend call") }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected host backend call") }
    fn alloc_from_host(&self, values: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        self.uploads.fetch_add(1, Ordering::SeqCst); Ok(Box::new(Buffer(values.to_vec())))
    }
    fn copy_to_host(&self, buffer: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        self.downloads.fetch_add(1, Ordering::SeqCst); Ok(buffer.as_any().downcast_ref::<Buffer>().unwrap().0.clone())
    }
    fn unary_dev(&self, kind: UnaryKind, input: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        self.kernels.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Buffer(CpuBackend.unary(kind, &input.as_any().downcast_ref::<Buffer>().unwrap().0))))
    }
}

#[test]
fn scalar_clamp_keeps_resident_forward_and_explicit_backward_host_fallback() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let backend = Arc::new(Counting::default()); register_backend(DEV, backend.clone());
    for (min, max) in [(f32::NAN, 1.), (-1., f32::NAN), (-0., 0.), (2., -2.), (f32::NEG_INFINITY, f32::INFINITY)] {
        let values = vec![f32::NEG_INFINITY, -1., -0., 0., 1., f32::INFINITY, f32::NAN];
        let x = Tensor::from_vec(values.clone(), &[7]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
        let before = backend.counts(); let output = x.clamp(min, max);
        assert_eq!(output.device(), DEV);
        let after = backend.counts(); assert_eq!([after[0] - before[0], after[1] - before[1], after[2] - before[2]], [0, 0, 1]);
        let expected = CpuBackend.unary(UnaryKind::Clamp { min, max }, &values);
        for (a, b) in output.to_vec().into_iter().zip(expected) { assert!(a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()); }
        let before = backend.counts();
        output.backward_with(&Tensor::full(&[7], f32::INFINITY));
        let after = backend.counts(); assert_eq!([after[0] - before[0], after[1] - before[1], after[2] - before[2]], [2, 2, 0]);
        let gradient = x.grad().unwrap(); assert_eq!(gradient.device(), DEV); assert!(!gradient.requires_grad());
        let expected = values.iter().map(|&x| if x >= min && x <= max { f32::INFINITY } else { 0. }).collect::<Vec<_>>();
        assert_eq!(gradient.to_vec(), expected);
    }
}
