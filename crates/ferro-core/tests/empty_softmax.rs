use ferro_core::dispatch::{register_backend, Backend, BinaryKind, DeviceBuffer, UnaryKind};
use ferro_core::{Device, Result, Tensor};
use std::any::Any;
use std::sync::Arc;

const DEV: Device = Device::Cuda(61);
struct Buffer(Vec<f32>);
impl DeviceBuffer for Buffer {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}
struct WithoutSoftmax;
impl Backend for WithoutSoftmax {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { unreachable!() }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { unreachable!() }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { unreachable!() }
    fn alloc_from_host(&self, x: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        Ok(Box::new(Buffer(x.to_vec())))
    }
    fn copy_to_host(&self, x: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        Ok(x.as_any().downcast_ref::<Buffer>().unwrap().0.clone())
    }
}

#[test]
fn empty_softmax_cpu_and_partial_backend_preserve_shape() {
    // This executable has one registry-using test and a private device ordinal.
    register_backend(DEV, Arc::new(WithoutSoftmax));
    for shape in [vec![0], vec![2, 0], vec![0, 3], vec![2, 0, 4]] {
        for device in [Device::Cpu, DEV] {
            let x = Tensor::from_vec(vec![], &shape).unwrap().to_device(device).unwrap();
            for dim in 0..shape.len() {
                let y = x.softmax(dim).unwrap();
                assert_eq!(y.shape(), shape);
                assert_eq!(y.device(), device);
                assert!(y.to_vec().is_empty());
                let log = x.log_softmax(dim).unwrap();
                assert_eq!(log.shape(), shape);
                assert!(log.to_vec().is_empty());
            }
            assert!(x.softmax(shape.len()).is_err());
            assert!(x.log_softmax(shape.len()).is_err());
        }
    }
}
