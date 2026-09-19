use ferro_core::{Device, Error, Result, Tensor};
use ferro_core::dispatch::{register_backend, Backend, BinaryKind, DeviceBuffer, SegmentOp, SegmentPlan, UnaryKind};
use ferro_core::segment::{self, PreparedSegments};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::any::Any;
const DEV: Device = Device::Cuda(27);
static LOCK: Mutex<()> = Mutex::new(());
struct Buffer(Vec<f32>);
impl DeviceBuffer for Buffer {
    fn as_any(&self) -> &dyn Any { self }
    fn len(&self) -> usize { self.0.len() }
    fn device(&self) -> Device { DEV }
}
struct Plan;
impl SegmentPlan for Plan { fn as_any(&self) -> &dyn Any { self } }
#[derive(Default)]
struct Fake { uploads: AtomicUsize, downloads: AtomicUsize, prepares: AtomicUsize, calls: AtomicUsize, fail: AtomicUsize }
fn failure() -> Error { Error::Unsupported { op: "injected_segment_launch", msg: "operational failure, not a fallback request".into() } }
impl Backend for Fake {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("host fallback") }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("host fallback") }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("host fallback") }
    fn alloc_from_host(&self, x: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        self.uploads.fetch_add(1, SeqCst); Ok(Box::new(Buffer(x.to_vec())))
    }
    fn copy_to_host(&self, x: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        self.downloads.fetch_add(1, SeqCst); Ok(x.as_any().downcast_ref::<Buffer>().unwrap().0.clone())
    }
    fn materialize_dev(&self, x: &dyn DeviceBuffer, shape: &[usize], strides: &[usize], offset: usize) -> Result<Box<dyn DeviceBuffer>> {
        if self.fail.load(SeqCst) != 0 { return Err(failure()); }
        let data = &x.as_any().downcast_ref::<Buffer>().unwrap().0;
        let out = (0..shape.iter().product()).map(|i| {
            let mut rem = i; let mut index = offset;
            for d in (0..shape.len()).rev() { index += rem % shape[d] * strides[d]; rem /= shape[d]; }
            data[index]
        }).collect();
        Ok(Box::new(Buffer(out)))
    }
    fn prepare_segments(&self, ids: &[usize], groups: usize) -> Result<Arc<dyn SegmentPlan>> {
        assert_eq!(ids, [0]); assert_eq!(groups, 1);
        self.prepares.fetch_add(1, SeqCst); Ok(Arc::new(Plan))
    }
    fn segment_dev(&self, _: &dyn SegmentPlan, op: SegmentOp, x: &dyn DeviceBuffer, _: Option<&dyn DeviceBuffer>, _: usize) -> Result<Box<dyn DeviceBuffer>> {
        self.calls.fetch_add(1, SeqCst);
        if self.fail.load(SeqCst) != 0 { return Err(failure()); }
        let data = &x.as_any().downcast_ref::<Buffer>().unwrap().0;
        let out = match op {
            SegmentOp::Sum | SegmentOp::SumBackward => data.clone(),
            SegmentOp::Softmax => vec![1.; data.len()],
            SegmentOp::SoftmaxBackward => vec![0.; data.len()],
        };
        Ok(Box::new(Buffer(out)))
    }
}

#[test]
fn one_preparation_and_no_feature_transfers_in_forward_backward() {
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let b = Arc::new(Fake::default()); register_backend(DEV, b.clone());
    let plan = PreparedSegments::new(&[0], 1, DEV).unwrap();
    for soft in [false, true] {
        let x = Tensor::from_vec(vec![2.,3.], &[1,2]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
        let seed = Tensor::from_vec(vec![4.,5.], &[1,2]).unwrap().to_device(DEV).unwrap();
        let uploads = b.uploads.load(SeqCst); let downloads = b.downloads.load(SeqCst);
        let y = if soft { plan.softmax(&x) } else { plan.sum(&x) }.unwrap();
        y.backward_with(&seed);
        assert_eq!(b.uploads.load(SeqCst), uploads);
        assert_eq!(b.downloads.load(SeqCst), downloads);
        assert_eq!(y.device(), DEV); assert_eq!(x.grad().unwrap().device(), DEV);
        assert_eq!(x.grad().unwrap().to_vec(), if soft { vec![0.,0.] } else { vec![4.,5.] });
    }
    assert_eq!(b.prepares.load(SeqCst), 1); assert_eq!(b.calls.load(SeqCst), 4);
}

struct Declining;
impl Backend for Declining {
    fn unary(&self, _: UnaryKind, _: &[f32]) -> Vec<f32> { panic!("host fallback") }
    fn binary(&self, _: BinaryKind, _: &[f32], _: &[f32]) -> Vec<f32> { panic!("host fallback") }
    fn matmul(&self, _: &[f32], _: &[f32], _: usize, _: usize, _: usize) -> Vec<f32> { panic!("host fallback") }
    fn alloc_from_host(&self, x: &[f32]) -> Result<Box<dyn DeviceBuffer>> { Ok(Box::new(Buffer(x.to_vec()))) }
    fn copy_to_host(&self, _: &dyn DeviceBuffer) -> Result<Vec<f32>> { panic!("declined segment must not download") }
}

#[test]
fn partial_backends_decline_explicitly_without_feature_download() {
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    register_backend(DEV, Arc::new(Declining));
    let x = Tensor::from_vec(vec![2.], &[1]).unwrap().to_device(DEV).unwrap();
    assert!(matches!(segment::sum(&x, &[0], 1), Err(Error::Unsupported { op: "prepare_segments", .. })));
    assert!(matches!(segment::softmax(&x, &[0], 1), Err(Error::Unsupported { op: "prepare_segments", .. })));
    let buffer = Buffer(vec![1.]);
    assert!(matches!(Declining.segment_dev(&Plan, SegmentOp::Sum, &buffer, None, 1), Err(Error::Unsupported { op: "segment_dev", .. })));
}

#[test]
fn failures_never_download_or_silently_fallback() {
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let b = Arc::new(Fake::default()); register_backend(DEV, b.clone());
    let x = Tensor::from_vec(vec![2.], &[1]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let p = PreparedSegments::new(&[0], 1, DEV).unwrap();
    let y = p.sum(&x).unwrap();
    b.fail.store(1, SeqCst);
    assert!(matches!(p.sum(&x), Err(Error::Unsupported { op: "injected_segment_launch", .. })));
    assert!(matches!(p.softmax(&x), Err(Error::Unsupported { op: "injected_segment_launch", .. })));
    let seed = Tensor::from_vec(vec![1.], &[1]).unwrap().to_device(DEV).unwrap();
    // Existing record_fn/backward API is infallible. Fail loudly, never recompute.
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| y.backward_with(&seed))).is_err());
    assert_eq!(b.downloads.load(SeqCst), 0);
    assert!(segment::sum(&x, &[1], 1).is_err());
    assert!(segment::mean(&x, &[0], 1).is_err());
    assert!(segment::max(&x, &[0], 1).is_err());
    ferro_core::capture(|| { assert!(p.sum(&x).is_err()); assert!(p.softmax(&x).is_err()); x.clone() });
}

#[test]
fn strided_seeds_and_reshape_adjoints_do_not_transfer() {
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let b = Arc::new(Fake::default()); register_backend(DEV, b.clone());
    let x = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], &[6]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], &[3,2]).unwrap().to_device(DEV).unwrap().transpose(0,1).unwrap();
    let before = (b.uploads.load(SeqCst), b.downloads.load(SeqCst));
    x.reshape(&[2,3]).unwrap().backward_with(&seed);
    assert_eq!((b.uploads.load(SeqCst), b.downloads.load(SeqCst)), before);
    assert_eq!(x.grad().unwrap().to_vec(), vec![1.,3.,5.,2.,4.,6.]);
}

#[test]
fn layout_operational_failure_is_not_silently_downloaded() {
    let _lock=LOCK.lock().unwrap_or_else(|e|e.into_inner());
    let b=Arc::new(Fake::default()); register_backend(DEV,b.clone());
    let x=Tensor::from_vec(vec![1.,2.,3.,4.], &[2,2]).unwrap().to_device(DEV).unwrap().transpose(0,1).unwrap();
    b.fail.store(1,SeqCst);
    assert!(matches!(x.reshape(&[4]), Err(Error::Unsupported { op: "injected_segment_launch", .. })));
    assert_eq!(b.downloads.load(SeqCst),0);
}
