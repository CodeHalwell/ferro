use std::any::Any;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use ferro_core::{capture, autograd::with_grad_enabled, DType, Device, Result, Tensor};
use ferro_core::dispatch::{Backend, BinaryKind, DeviceBuffer, UnaryKind, register_backend};

#[test]
fn compatible_stride_chunks_share_typed_storage_and_incompatible_chunks_reject() {
    for dtype in [DType::F32, DType::F64, DType::I64, DType::Bool, DType::F16, DType::BF16] {
        let base = Tensor::from_vec((0..12).map(|n| n as f32).collect(), &[2, 6]).unwrap().to_dtype(dtype);
        let stepped = base.slice(1, 1, 6, 2).unwrap();
        let alias = stepped.view(&[3, 2]).unwrap();
        assert_eq!(alias.strides(), &[4, 2]);
        assert_eq!(alias.storage_offset(), 1);
        assert_eq!(alias._storage_ptr(), base._storage_ptr());
        assert_eq!(alias.to_vec_f64(), stepped.to_vec_f64());
        let transposed = base.transpose(0, 1).unwrap();
        assert!(transposed.view(&[12]).is_err());
        assert_eq!(transposed.view(&[3, 2, 2]).unwrap().strides(), &[2, 1, 6]);
        base.copy_from(&Tensor::zeros(base.shape()).to_dtype(dtype)).unwrap();
        assert_eq!(alias.to_vec_f64(), vec![0.; 6]);
        assert_eq!(alias._version(), base._version());
        assert_eq!(alias.view_as(&Tensor::zeros(&[6])).unwrap().shape(), &[6]);
    }
    let repeated = Tensor::ones(&[1, 3]).expand(&[2, 3]).unwrap();
    assert!(repeated.view(&[6]).is_err());
    assert_eq!(repeated.view(&[2, 1, 3]).unwrap().strides(), &[0, 3, 1]);
    let scalar = Tensor::scalar(2.).expand(&[2, 3]).unwrap();
    assert_eq!(scalar.view(&[6]).unwrap().strides(), &[0]);
    let singleton = Tensor::ones(&[2, 1]).transpose(0, 1).unwrap();
    assert_eq!(singleton.view(&[1, 2]).unwrap().strides(), &[2, 1]);
}

#[test]
fn empty_scalar_and_large_virtual_views_preserve_offsets_without_allocating_data() {
    let empty = Tensor::zeros(&[0, 8]).narrow(1, 4, 4).unwrap();
    assert_eq!(empty.view(empty.shape()).unwrap().strides(), empty.strides());
    let changed = empty.view(&[2, 0, 2]).unwrap();
    assert_eq!(changed.strides(), &[2, 2, 1]);
    assert_eq!(changed.storage_offset(), 4);
    assert_eq!(changed._storage_ptr(), empty._storage_ptr());
    assert!(changed.to_vec().is_empty());
    let scalar = Tensor::scalar(2.);
    assert_eq!(scalar.view(&[1, 1]).unwrap().strides(), &[1, 1]);
    assert_eq!(scalar.view(&[]).unwrap().item(), 2.);
    let huge = scalar.expand(&[isize::MAX as usize / 4 + 1]).unwrap();
    assert_eq!(huge.view(&[1, huge.numel()]).unwrap().strides(), &[0, 0]);
    assert!(empty.view(&[usize::MAX, 2, 0]).is_err());
    assert!(scalar.view(&[2]).is_err());
    assert_eq!(scalar.item(), 2.);
}

#[test]
fn view_adjoints_keep_metadata_flags_saved_value_guards_and_live_replay() {
    for dtype in [DType::F32, DType::F64] {
        let x = Tensor::from_vec((0..12).map(|n| n as f32).collect(), &[2, 6]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let y = x.slice(1, 1, 6, 2).unwrap().view(&[3, 2]).unwrap();
        let gradient = y.square().unwrap().sum().grad_wrt(&[&x], false).unwrap().remove(0);
        assert_eq!(gradient.to_vec_f64(), vec![0., 2., 0., 6., 0., 10., 0., 14., 0., 18., 0., 22.]);
        let blocked = with_grad_enabled(false, || y.view(&[6])).unwrap();
        assert!(blocked.requires_grad() && blocked.is_leaf());
        let stale = y.square().unwrap().sum();
        with_grad_enabled(false, || x.copy_leaf_from(&Tensor::zeros(x.shape()).to_dtype(dtype))).unwrap();
        assert!(stale.grad_wrt(&[&x], false).is_err());
        assert_eq!(y.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec_f64(), vec![0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]);
        let source = Tensor::from_vec((0..12).map(|n| n as f32).collect(), &[2, 6]).unwrap().to_dtype(dtype);
        let layout = source.slice(1, 1, 6, 2).unwrap();
        let traced = capture(|| layout.view(&[3, 2])).unwrap();
        let chain = ferro_core::graph::CompiledChain::compile(&traced).unwrap();
        source.copy_from(&Tensor::ones(source.shape()).to_dtype(dtype)).unwrap();
        let replay = chain.replay().unwrap();
        assert_eq!(replay.strides(), &[4, 2]);
        assert_eq!(replay._storage_ptr(), source._storage_ptr());
        assert_eq!(replay.to_vec_f64(), vec![1.; 6]);
    }
    let x = Tensor::ones(&[2, 3]).requires_grad_(true).unwrap();
    let y = x.view(&[6]).unwrap().mul(&x.view(&[6]).unwrap()).unwrap().sum();
    let first = y.grad_wrt(&[&x], true).unwrap().remove(0);
    assert_eq!(first.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec(), vec![2.; 6]);
    let direction = Tensor::from_vec((0..6).map(|n| n as f32).collect(), &[3, 2]).unwrap().transpose(0, 1).unwrap();
    let level = ferro_core::forward_ad::DualLevel::new(&[&x], &[&direction]).unwrap();
    let output = x.view(&[6]).unwrap();
    assert_eq!(level.tangent(&output).unwrap().unwrap().to_vec(), vec![0., 2., 4., 1., 3., 5.]);
}

const DEV: Device = Device::Cuda(246);
static SERIAL: Mutex<()> = Mutex::new(());
struct Buffer(Vec<f32>);
impl DeviceBuffer for Buffer {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}
#[derive(Default)]
struct Counting { uploads: AtomicUsize, downloads: AtomicUsize, materializations: AtomicUsize, copies: AtomicUsize }
impl Backend for Counting {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("unexpected host kernel") }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("unexpected host kernel") }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("unexpected host kernel") }
    fn alloc_from_host(&self, data: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        self.uploads.fetch_add(1, Ordering::SeqCst); Ok(Box::new(Buffer(data.to_vec())))
    }
    fn copy_to_host(&self, data: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        self.downloads.fetch_add(1, Ordering::SeqCst); Ok(data.as_any().downcast_ref::<Buffer>().unwrap().0.clone())
    }
    fn copy_dev(&self, data: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        self.copies.fetch_add(1, Ordering::SeqCst); Ok(Box::new(Buffer(data.as_any().downcast_ref::<Buffer>().unwrap().0.clone())))
    }
    fn materialize_dev(&self, _: &dyn DeviceBuffer, _: &[usize], _: &[usize], _: usize) -> Result<Box<dyn DeviceBuffer>> {
        self.materializations.fetch_add(1, Ordering::SeqCst); panic!("view must never materialize")
    }
}

#[test]
fn device_views_and_rejections_perform_no_transfers_or_materialization() {
    let _lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let backend = Arc::new(Counting::default()); register_backend(DEV, backend.clone());
    let base = Tensor::from_vec((0..12).map(|n| n as f32).collect(), &[2, 6]).unwrap().to_device(DEV).unwrap();
    let stepped = base.slice(1, 1, 6, 2).unwrap();
    let output = stepped.view(&[3, 2]).unwrap();
    assert_eq!(output._storage_ptr(), base._storage_ptr());
    assert_eq!(output.storage_offset(), 1);
    assert!(base.transpose(0, 1).unwrap().view(&[12]).is_err());
    assert_eq!(backend.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(backend.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(backend.materializations.load(Ordering::SeqCst), 0);
    assert_eq!(backend.copies.load(Ordering::SeqCst), 0);
    // Recording device leaves retains the established isolated snapshot policy.
    let traced = capture(|| base.view(&[3, 4])).unwrap();
    assert_ne!(traced._storage_ptr(), base._storage_ptr());
    assert_eq!(backend.copies.load(Ordering::SeqCst), 1);
    let chain = ferro_core::graph::CompiledChain::compile(&traced).unwrap();
    for _ in 0..2 {
        let replay = chain.replay().unwrap();
        assert_eq!(replay._storage_ptr(), base._storage_ptr());
        assert_eq!(replay.strides(), &[4, 1]);
    }
    assert_eq!(backend.uploads.load(Ordering::SeqCst), 1);
    assert_eq!(backend.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(backend.materializations.load(Ordering::SeqCst), 0);
    assert_eq!(backend.copies.load(Ordering::SeqCst), 1);
}
