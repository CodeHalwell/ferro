use ferro_core::{Device, Error, Result, Tensor};
use ferro_core::dispatch::{register_backend, Backend, BinaryKind, DeviceBuffer, UnaryKind};
use std::any::Any;
use std::sync::{Arc, Mutex};

const DEV: Device = Device::Cuda(87);
static LOCK: Mutex<()> = Mutex::new(());
struct Buffer<T>(Vec<T>);
impl<T: Send + Sync + 'static> DeviceBuffer for Buffer<T> {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}
struct Partial;
struct Fault;
impl Backend for Fault {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { unreachable!() }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { unreachable!() }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { unreachable!() }
    fn alloc_from_host(&self, v: &[f32]) -> Result<Box<dyn DeviceBuffer>> { Partial.alloc_from_host(v) }
    fn copy_to_host(&self, _: &dyn DeviceBuffer) -> Result<Vec<f32>> { panic!("operational failure must not download") }
    fn reduce_dev(&self, _: ferro_core::dispatch::ReduceKind, _: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        Err(Error::Unsupported { op: "reduce_dev", msg: "injected launch failure".into() })
    }
    fn sum_dim_dev(&self, _: &dyn DeviceBuffer, _: &[usize], _: usize) -> Result<Box<dyn DeviceBuffer>> {
        Err(Error::Io { op: "sum_dim_dev", msg: "injected launch failure".into() })
    }
}

#[test]
fn operational_reduction_errors_are_not_host_fallbacks() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(Fault));
    let t = Tensor::ones(&[2, 3]).to_device(DEV).unwrap();
    assert!(matches!(t.sum_dim(1, false), Err(Error::Io { op: "sum_dim_dev", msg }) if msg == "injected launch failure"));
    for op in [0, 1] {
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match op { 0 => { t.sum(); }, _ => { t.mean(); } }
        })).expect_err("infallible wrappers must expose operational failures");
        let msg = panic.downcast_ref::<String>().map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied()).unwrap();
        assert!(msg.contains("injected launch failure"), "{msg}");
    }
}


#[test]
fn partial_backend_reductions_fall_back_with_gradients() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(Partial));
    let t = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let rows = t.sum_dim(1, false).unwrap();
    assert_eq!(rows.device(), Device::Cpu);
    assert_eq!(rows.to_vec(), vec![6., 15.]);
    rows.sum().backward();
    assert_eq!(t.grad().unwrap().device(), DEV);
    assert_eq!(t.grad().unwrap().to_vec(), vec![1.; 6]);
    assert_eq!(t.sum().item(), 21.);
    assert_eq!(t.mean().item(), 3.5);
    assert_eq!(t.sum_dim(0, true).unwrap().to_vec(), vec![5., 7., 9.]);
    let bias = Tensor::from_vec(vec![1., 2., 3.], &[3]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::ones(&[2, 3]).to_device(DEV).unwrap();
    t.add(&bias).unwrap().backward_with(&seed);
    assert_eq!(bias.grad().unwrap().device(), DEV);
    assert_eq!(bias.grad().unwrap().to_vec(), vec![2.; 3]);
}

impl Backend for Partial {
    fn binary_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        self.binary_bc_dev(kind, a, &[a.len()], b, &[b.len()], &[a.len()])
    }
    fn binary_bc_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, sa: &[usize], b: &dyn DeviceBuffer, sb: &[usize], _: &[usize]) -> Result<Box<dyn DeviceBuffer>> {
        let a = Tensor::from_vec(self.copy_to_host(a)?, sa)?;
        let b = Tensor::from_vec(self.copy_to_host(b)?, sb)?;
        let out = match kind { BinaryKind::Add => a.add(&b)?, BinaryKind::Sub => a.sub(&b)?, BinaryKind::Mul => a.mul(&b)?, BinaryKind::Div => a.div(&b)? };
        self.alloc_from_host(&out.to_vec())
    }
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { unreachable!() }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { unreachable!() }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { unreachable!() }
    fn alloc_from_host(&self, v: &[f32]) -> Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buffer(v.to_vec()))) }
    fn copy_to_host(&self, b: &dyn DeviceBuffer) -> Result<Vec<f32>> { Ok(b.as_any().downcast_ref::<Buffer<f32>>().unwrap().0.clone()) }
    fn alloc_i64_from_host(&self, v: &[i64]) -> Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buffer(v.to_vec()))) }
    fn copy_i64_to_host(&self, b: &dyn DeviceBuffer) -> Result<Vec<i64>> { Ok(b.as_any().downcast_ref::<Buffer<i64>>().unwrap().0.clone()) }
}

#[test]
fn i64_device_views_follow_logical_order_without_float_roundtrip() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(Partial));
    let values = vec![i64::MAX, 16_777_217, -9_007_199_254_740_993, 4, 5, i64::MIN];
    let expected = vec![values[0], values[3], values[1], values[4], values[2], values[5]];
    let t = Tensor::from_vec_i64(values, &[2, 3]).unwrap().to_device(DEV).unwrap().transpose(0, 1).unwrap();
    assert_eq!(t.to_vec_i64(), expected);
    assert_eq!(t.to_vec(), expected.iter().map(|&x| x as f32).collect::<Vec<_>>());
    assert_eq!(t.to_vec_f64(), expected.iter().map(|&x| x as f64).collect::<Vec<_>>());
    assert_eq!(t.to_device(Device::Cpu).unwrap().to_vec_i64(), expected);
    let flat = t.reshape(&[6]).unwrap();
    assert_eq!(flat.device(), DEV);
    assert_eq!(flat.to_vec_i64(), expected);
    assert_eq!(t.detach_copy().to_vec_i64(), expected);
}


#[test]
fn checked_shape_boundaries() {
    for shape in [&[usize::MAX, 2][..], &[0, usize::MAX, 2], &[usize::MAX, 0, 2], &[usize::MAX, 2, 0]] {
        assert!(matches!(Tensor::from_vec(vec![], shape), Err(Error::InvalidShape { .. })));
        assert!(matches!(Tensor::from_vec_i64(vec![], shape), Err(Error::InvalidShape { .. })));
        assert!(matches!(Tensor::full_on(shape, 0., Device::Cpu), Err(Error::InvalidShape { .. })));
        assert!(matches!(Tensor::full_on(shape, 0., DEV), Err(Error::InvalidShape { .. })));
        let empty = Tensor::from_vec(vec![], &[0]).unwrap();
        assert!(matches!(empty.reshape(shape), Err(Error::InvalidShape { .. })));
    }
    for shape in [&[0, 3][..], &[3, 0], &[2, 0, 3]] {
        let t = Tensor::from_vec(vec![], shape).unwrap();
        assert_eq!(t.numel(), 0);
        assert_eq!(t.transpose(0, 1).unwrap().to_vec(), Vec::<f32>::new());
    }
    assert_eq!(Tensor::from_vec(vec![1.], &[]).unwrap().numel(), 1);
}
