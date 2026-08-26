//! Device-side semantics of the public in-place API over a minimal fake
//! resident backend: mutation lands in the original device buffer through
//! the *_inplace_dev kernels, and the uniqueness gate refuses shared device
//! storage - device detach_copy shares buffers with backward-closure
//! snapshots, so an aliased device tensor is exactly the thing a mutation
//! could silently poison.

use std::any::Any;
use std::sync::{Arc, Mutex};

use ferro_core::dispatch::{register_backend, Backend, BinaryKind, DeviceBuffer, UnaryKind};
use ferro_core::{Device, Result, Tensor};

const DEV: Device = Device::Cuda(9);

struct Buf(Mutex<Vec<f32>>);

impl DeviceBuffer for Buf {
    fn device(&self) -> Device {
        DEV
    }
    fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn cell(b: &dyn DeviceBuffer) -> &Mutex<Vec<f32>> {
    &b.as_any().downcast_ref::<Buf>().expect("foreign buffer").0
}

struct Fake;

impl Backend for Fake {
    fn unary(&self, _k: UnaryKind, _x: &[f32]) -> Vec<f32> {
        panic!("host path must not run");
    }
    fn binary(&self, _k: BinaryKind, _a: &[f32], _b: &[f32]) -> Vec<f32> {
        panic!("host path must not run");
    }
    fn matmul(&self, _a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        panic!("host path must not run");
    }
    fn alloc_from_host(&self, d: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        Ok(Box::new(Buf(Mutex::new(d.to_vec()))))
    }
    fn copy_to_host(&self, b: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        Ok(cell(b).lock().unwrap().clone())
    }
    fn write_dev_from_host(&self, dst: &dyn DeviceBuffer, d: &[f32]) -> Result<()> {
        cell(dst).lock().unwrap().copy_from_slice(d);
        Ok(())
    }
    fn copy_into_dev(&self, dst: &dyn DeviceBuffer, src: &dyn DeviceBuffer) -> Result<()> {
        let s = cell(src).lock().unwrap().clone();
        cell(dst).lock().unwrap().copy_from_slice(&s);
        Ok(())
    }
    fn affine_inplace_dev(&self, dst: &dyn DeviceBuffer, mul: f32, add: f32) -> Result<()> {
        self.affine_inplace(&mut cell(dst).lock().unwrap(), mul, add);
        Ok(())
    }
    fn binary_inplace_dev(
        &self,
        kind: BinaryKind,
        dst: &dyn DeviceBuffer,
        src: &dyn DeviceBuffer,
    ) -> Result<()> {
        let s = cell(src).lock().unwrap().clone(); // dst may alias src
        self.binary_inplace(kind, &mut cell(dst).lock().unwrap(), &s);
        Ok(())
    }
}

fn setup() {
    register_backend(DEV, Arc::new(Fake));
}

fn dev(v: &[f32]) -> Tensor {
    Tensor::from_vec(v.to_vec(), &[v.len()])
        .unwrap()
        .to_device(DEV)
        .unwrap()
}

#[test]
fn device_mutation_lands_in_the_original_buffer() {
    setup();
    let x = dev(&[1.0, 2.0, 3.0]);
    x.mul_scalar_(2.0).unwrap();
    x.add_(&dev(&[1.0, 1.0, 1.0])).unwrap();
    assert_eq!(x.device(), DEV, "mutation kept the tensor resident");
    assert_eq!(x.to_vec(), vec![3.0, 5.0, 7.0]);
    assert_eq!(x._version(), 2);
    // Self-aliased op through the same-index kernel contract.
    x.add_(&x).unwrap();
    assert_eq!(x.to_vec(), vec![6.0, 10.0, 14.0]);
}

#[test]
fn shared_device_storage_is_refused() {
    setup();
    let x = dev(&[1.0, 2.0]);
    // Device detach_copy shares the buffer (that is its documented perf
    // contract) - so neither alias may be mutated while both exist.
    let alias = x.detach_copy();
    assert!(alias.mul_scalar_(2.0).is_err());
    assert!(x.mul_scalar_(2.0).is_err());
    drop(alias);
    x.mul_scalar_(2.0).unwrap();
    assert_eq!(x.to_vec(), vec![2.0, 4.0]);
}

#[test]
fn copy_from_moves_data_across_devices_into_stable_storage() {
    setup();
    let x = dev(&[0.0, 0.0]);
    let ptr = x._storage_ptr();

    // host -> device lands via write_dev_from_host.
    x.copy_from(&Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap())
        .unwrap();
    assert_eq!(x.to_vec(), vec![1.0, 2.0]);

    // device -> device via copy_into_dev.
    x.copy_from(&dev(&[5.0, 6.0])).unwrap();
    assert_eq!(x.to_vec(), vec![5.0, 6.0]);

    // device -> host dst downloads through the ordinary materialize path.
    let h = Tensor::from_vec(vec![0.0, 0.0], &[2]).unwrap();
    h.copy_from(&x).unwrap();
    assert_eq!(h.to_vec(), vec![5.0, 6.0]);

    assert_eq!(x._storage_ptr(), ptr, "copy_from preserved the allocation");
    assert_eq!(x.device(), DEV);
}

#[test]
fn cross_device_elementwise_is_refused() {
    setup();
    let x = dev(&[1.0, 2.0]);
    let h = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    assert!(x.add_(&h).is_err(), "device dst, host src");
    assert!(h.add_(&x).is_err(), "host dst, device src");
    assert_eq!(x.to_vec(), vec![1.0, 2.0]);
    assert_eq!(h.to_vec(), vec![1.0, 2.0]);
}
