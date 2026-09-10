use std::any::Any;
use std::sync::{Arc, Mutex};
use ferro_core::{capture, Device, Error, Result, Tensor};
use ferro_core::dispatch::{register_backend, Backend, BinaryKind, CpuBackend, DeviceBuffer, UnaryKind};
use ferro_core::graph::CompiledChain;

const DEV: Device = Device::Cuda(29);
static LOCK: Mutex<()> = Mutex::new(());
struct Buf(Mutex<Vec<f32>>);
fn buffer(v: Vec<f32>) -> Box<dyn DeviceBuffer> { Box::new(Buf(Mutex::new(v))) }
impl DeviceBuffer for Buf {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.lock().unwrap().len() }
    fn as_any(&self) -> &dyn Any { self }
}
fn data(x: &dyn DeviceBuffer) -> Vec<f32> { x.as_any().downcast_ref::<Buf>().unwrap().0.lock().unwrap().clone() }
struct Partial(&'static str);
impl Partial {
    fn check(&self, op: &'static str) -> Result<()> {
        if self.0 == op || self.0 == "all" { Err(Error::Unsupported { op, msg: "declined kernel".into() }) } else { Ok(()) }
    }
}
impl Backend for Partial {
    fn unary(&self, k: UnaryKind, x: &[f32]) -> Vec<f32> { CpuBackend.unary(k, x) }
    fn binary(&self, k: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> { CpuBackend.binary(k, a, b) }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> { CpuBackend.matmul(a, b, m, k, n) }
    fn alloc_from_host(&self, x: &[f32]) -> Result<Box<dyn DeviceBuffer>> { Ok(buffer(x.to_vec())) }
    fn copy_to_host(&self, x: &dyn DeviceBuffer) -> Result<Vec<f32>> { Ok(data(x).to_vec()) }
    fn fill_dev(&self, v: f32, n: usize) -> Result<Box<dyn DeviceBuffer>> { Ok(buffer(vec![v; n])) }
    fn write_dev_from_host(&self, dst: &dyn DeviceBuffer, src: &[f32]) -> Result<()> {
        dst.as_any().downcast_ref::<Buf>().unwrap().0.lock().unwrap().copy_from_slice(src);
        Ok(())
    }
    fn sum_dim_dev(&self, x: &dyn DeviceBuffer, s: &[usize], d: usize) -> Result<Box<dyn DeviceBuffer>> {
        self.check("reduction")?;
        Ok(buffer(Tensor::from_vec(data(x), s)?.sum_dim(d, true)?.to_vec()))
    }
    fn unary_dev(&self, k: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        self.check("unary")?;
        Ok(buffer(CpuBackend.unary(k, &data(x))))
    }
    fn binary_dev(&self, k: BinaryKind, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        self.check("binary")?;
        Ok(buffer(CpuBackend.binary(k, &data(a), &data(b))))
    }
    fn binary_bc_dev(&self, k: BinaryKind, a: &dyn DeviceBuffer, sa: &[usize], b: &dyn DeviceBuffer, sb: &[usize], _: &[usize]) -> Result<Box<dyn DeviceBuffer>> {
        self.check("broadcast")?;
        let a = Tensor::from_vec(data(a).to_vec(), sa)?;
        let b = Tensor::from_vec(data(b).to_vec(), sb)?;
        let out = match k { BinaryKind::Add => a.add(&b), BinaryKind::Sub => a.sub(&b), BinaryKind::Mul => a.mul(&b), BinaryKind::Div => a.div(&b) }?;
        Ok(buffer(out.to_vec()))
    }
}
fn close(a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape(), b.shape());
    for (a, b) in a.to_vec().iter().zip(b.to_vec()) { assert!((a-b).abs() < 2e-5, "{a} != {b}"); }
}
fn fallback(decline: &'static str) {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(Partial(decline)));
    let host = Tensor::from_vec(vec![1., 2., 4., -2., 3., 1.], &[2, 3]).unwrap();
    let hw = Tensor::from_vec(vec![0.5, 1.5, -0.7], &[3]).unwrap();
    let hb = Tensor::from_vec(vec![0.2, -0.3, 0.8], &[3]).unwrap();
    let x = host.to_device(DEV).unwrap();
    let w = hw.to_device(DEV).unwrap();
    let b = hb.to_device(DEV).unwrap();
    let out = capture(|| x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap());
    assert_eq!(out.device(), Device::Cpu);
    close(&out, &host.layer_norm(Some(&hw), Some(&hb), 1e-5).unwrap());
    let compiled = CompiledChain::compile(&out).unwrap();
    x.fill_(9.).unwrap();
    let fresh = x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap();
    assert_ne!(fresh.to_vec(), out.to_vec());
    close(&compiled.replay().unwrap(), &fresh);
    let x = host.to_device(DEV).unwrap();
    let host = host.requires_grad_(true).unwrap();
    let hw = hw.requires_grad_(true).unwrap();
    let hb = hb.requires_grad_(true).unwrap();
    let x = x.requires_grad_(true).unwrap();
    let w = w.requires_grad_(true).unwrap();
    let b = b.requires_grad_(true).unwrap();
    let coefficients = Tensor::from_vec(vec![1., -0.4, 0.7, -0.2, 1.3, 0.8], &[2, 3]).unwrap();
    x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap().mul(&coefficients).unwrap().sum().backward();
    host.layer_norm(Some(&hw), Some(&hb), 1e-5).unwrap().mul(&coefficients).unwrap().sum().backward();
    for (a, b) in [(&x, &host), (&w, &hw), (&b, &hb)] {
        let grad = a.grad().unwrap();
        assert_eq!(grad.device(), DEV);
        close(&grad, &b.grad().unwrap());
    }
}
#[test] fn layer_norm_all_kernels_declined() { fallback("all"); }
#[test] fn layer_norm_reduction_declined() { fallback("reduction"); }
#[test] fn layer_norm_broadcast_declined() { fallback("broadcast"); }
#[test] fn layer_norm_unary_declined() { fallback("unary"); }
#[test] fn layer_norm_binary_declined() { fallback("binary"); }

#[test]
fn layer_norm_strided_input_without_layout_kernels() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(Partial("layout")));
    let host = Tensor::from_vec(vec![1., 2., 4., -2., 3., 1.], &[3, 2]).unwrap();
    let x = host.to_device(DEV).unwrap();
    let out = capture(|| x.transpose(0, 1).unwrap().layer_norm(None, None, 1e-5).unwrap());
    assert_eq!(out.device(), Device::Cpu);
    close(&out, &host.transpose(0, 1).unwrap().layer_norm(None, None, 1e-5).unwrap());
    let compiled = CompiledChain::compile(&out).unwrap();
    close(&compiled.replay().unwrap(), &out);
}
