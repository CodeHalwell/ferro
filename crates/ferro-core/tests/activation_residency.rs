use std::any::Any;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use ferro_core::{Device, Result, Tensor};
use ferro_core::dispatch::{register_backend, Backend, BinaryKind, CpuBackend, DeviceBuffer, UnaryKind};

const DEV: Device = Device::Cuda(47);
static SERIAL: Mutex<()> = Mutex::new(());
struct Buf(Vec<f32>);
impl DeviceBuffer for Buf {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}
fn values(buffer: &dyn DeviceBuffer) -> &[f32] { &buffer.as_any().downcast_ref::<Buf>().unwrap().0 }
fn buffer(values: Vec<f32>) -> Box<dyn DeviceBuffer> { Box::new(Buf(values)) }
#[derive(Default)]
struct Counting { uploads: AtomicUsize, downloads: AtomicUsize, binaries: AtomicUsize, unaries: AtomicUsize }
impl Backend for Counting {
    fn prepare_static(self: Arc<Self>, runs: &[ferro_core::dispatch::StaticRun], _leaves: Vec<Arc<dyn DeviceBuffer>>) -> Result<Box<dyn ferro_core::dispatch::StaticExecution>> {
        assert_eq!(runs.len(),1);
        assert!(matches!(&runs[0].op, ferro_core::dispatch::StaticOp::Pointwise(steps) if matches!(steps.as_slice(), [ferro_core::dispatch::ChainStepRef::Unary(UnaryKind::Powf(p))] if *p==0.5)));
        Err(ferro_core::Error::Unsupported { op: "counted_static_power", msg: "translated exact half-power".into() })
    }
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> { CpuBackend.unary(kind,x) }
    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> { CpuBackend.binary(kind,a,b) }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> { CpuBackend.matmul(a,b,m,k,n) }
    fn alloc_from_host(&self, x: &[f32]) -> Result<Box<dyn DeviceBuffer>> { self.uploads.fetch_add(1,Ordering::SeqCst); Ok(buffer(x.to_vec())) }
    fn copy_to_host(&self, x: &dyn DeviceBuffer) -> Result<Vec<f32>> { self.downloads.fetch_add(1,Ordering::SeqCst); Ok(values(x).to_vec()) }
    fn copy_dev(&self, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> { Ok(buffer(values(x).to_vec())) }
    fn fill_dev(&self, value: f32, count: usize) -> Result<Box<dyn DeviceBuffer>> { Ok(buffer(vec![value;count])) }
    fn unary_dev(&self, kind: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        self.unaries.fetch_add(1,Ordering::SeqCst); Ok(buffer(CpuBackend.unary(kind,values(x))))
    }
    fn binary_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        self.binaries.fetch_add(1,Ordering::SeqCst); Ok(buffer(CpuBackend.binary(kind,values(a),values(b))))
    }
    fn binary_bc_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, sa: &[usize], b: &dyn DeviceBuffer, sb: &[usize], output: &[usize]) -> Result<Box<dyn DeviceBuffer>> {
        self.binaries.fetch_add(1,Ordering::SeqCst); Ok(buffer(CpuBackend.binary_bc(kind,values(a),sa,values(b),sb,output)))
    }
}

#[test]
fn higher_activation_callbacks_keep_parent_buffers_resident_without_snapshot_transfers() {
    let _guard=SERIAL.lock().unwrap_or_else(|e|e.into_inner());
    let backend=Arc::new(Counting::default());register_backend(DEV,backend.clone());
    for tanh in [false,true] {
        let x=Tensor::from_vec(vec![0.2,-0.7,1.3],&[3]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
        let seed=Tensor::full_on(&[3],0.7,DEV).unwrap().requires_grad_(true).unwrap();
        let second_seed=Tensor::full_on(&[3],1.,DEV).unwrap();
        let uploads=backend.uploads.load(Ordering::SeqCst);let downloads=backend.downloads.load(Ordering::SeqCst);
        let output=if tanh{x.tanh()}else{x.sigmoid()};
        let first=output.vjp_wrt(&[&x],&seed,true).unwrap().remove(0);
        let second=first.vjp_wrt(&[&x,&seed],&second_seed,false).unwrap();
        assert_eq!(first.device(),DEV);assert!(second.iter().all(|g|g.device()==DEV));
        assert_eq!(backend.uploads.load(Ordering::SeqCst),uploads);assert_eq!(backend.downloads.load(Ordering::SeqCst),downloads);
        assert!(x.grad().is_none()&&seed.grad().is_none());
    }
    assert!(backend.binaries.load(Ordering::SeqCst)>8);
}

#[test]
fn activation_saved_output_backward_uses_resident_f32_primitives_without_host_transfers() {
    let _guard=SERIAL.lock().unwrap_or_else(|e|e.into_inner());
    let backend=Arc::new(Counting::default());
    register_backend(DEV,backend.clone());
    for kind in 0..3 {
        let x=Tensor::from_vec(vec![0.2,-0.7,1.3],&[3]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
        let seed=Tensor::from_vec(vec![0.7,-0.3,0.2],&[3]).unwrap().to_device(DEV).unwrap();
        let uploads=backend.uploads.load(Ordering::SeqCst);
        let downloads=backend.downloads.load(Ordering::SeqCst);
        let out=match kind {0=>x.exp(),1=>x.sigmoid(),_=>x.tanh()};
        out.backward_with(&seed);
        let grad=x.grad().unwrap();
        assert_eq!(out.device(),DEV); assert_eq!(grad.device(),DEV);
        assert_eq!(backend.uploads.load(Ordering::SeqCst),uploads);
        assert_eq!(backend.downloads.load(Ordering::SeqCst),downloads);
        let output=out.to_vec();
        for ((actual,y),g) in grad.to_vec().into_iter().zip(output).zip([0.7,-0.3,0.2]) {
            let expected=match kind {0=>g*y,1=>(g*(1.-y))*y,_=>g*(1.-y*y)};
            assert!((actual-expected).abs()<2e-7);
        }
    }
    assert_eq!(backend.unaries.load(Ordering::SeqCst),3);
    assert!(backend.binaries.load(Ordering::SeqCst)>=7);
}

#[test]
fn output_saved_extended_unaries_keep_explicit_host_fallback_and_device_gradients() {
    let _guard=SERIAL.lock().unwrap_or_else(|e|e.into_inner());
    let backend=Arc::new(Counting::default());register_backend(DEV,backend.clone());
    for kind in 0..4 {
        let x=Tensor::from_vec(vec![0.2,0.7,1.3],&[3]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
        let seed=Tensor::full_on(&[3],0.3,DEV).unwrap();
        let (uploads,downloads)=(backend.uploads.load(Ordering::SeqCst),backend.downloads.load(Ordering::SeqCst));
        let output=match kind{0=>x.expm1().unwrap(),1=>x.exp2().unwrap(),2=>x.reciprocal().unwrap(),_=>x.tan().unwrap()};
        assert_eq!(output.device(),DEV);
        assert_eq!(backend.uploads.load(Ordering::SeqCst)-uploads,1);
        assert_eq!(backend.downloads.load(Ordering::SeqCst)-downloads,2);
        let grad=output.vjp_wrt(&[&x],&seed,false).unwrap().remove(0);assert_eq!(grad.device(),DEV);
        assert_eq!(backend.uploads.load(Ordering::SeqCst)-uploads,2);
        assert_eq!(backend.downloads.load(Ordering::SeqCst)-downloads,4);
        let expected=output.to_vec().into_iter().map(|y|match kind{0=>0.3*(y+1.),1=>(0.3*y)*std::f32::consts::LN_2,2=>(-0.3)*(y*y),_=>0.3*(y*y+1.)}).collect::<Vec<_>>();
        assert_eq!(grad.to_vec(),expected);assert!(!grad.requires_grad());assert!(x.grad().is_none());
    }
    assert_eq!(backend.unaries.load(Ordering::SeqCst),0);assert_eq!(backend.binaries.load(Ordering::SeqCst),0);
}

#[test]
fn half_power_device_forward_preserves_existing_explicit_adjoint_host_fallback() {
    let _guard=SERIAL.lock().unwrap_or_else(|e|e.into_inner());
    let backend=Arc::new(Counting::default());register_backend(DEV,backend.clone());
    for p in [0.5,-0.5] {
        let x=Tensor::from_vec(vec![0.25,4.,9.],&[3]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
        let seed=Tensor::full_on(&[3],1.,DEV).unwrap();
        let (up,down,unary)=(backend.uploads.load(Ordering::SeqCst),backend.downloads.load(Ordering::SeqCst),backend.unaries.load(Ordering::SeqCst));
        let y=x.powf(p);assert_eq!(y.device(),DEV);assert_eq!(backend.unaries.load(Ordering::SeqCst)-unary,1);
        assert_eq!(backend.uploads.load(Ordering::SeqCst),up);assert_eq!(backend.downloads.load(Ordering::SeqCst),down);
        let g=y.vjp_wrt(&[&x],&seed,false).unwrap().remove(0);assert_eq!(g.device(),DEV);
        assert_eq!(backend.uploads.load(Ordering::SeqCst)-up,1);assert_eq!(backend.downloads.load(Ordering::SeqCst)-down,2);
        assert_eq!(g.to_vec(),[0.25f32,4.,9.].into_iter().map(|x|p*if p==0.5{x.sqrt().recip()}else{x.powf(p-1.)}).collect::<Vec<_>>());
        assert!(!g.requires_grad());assert!(x.grad().is_none());
    }
}

#[test]
fn original_scalar_power_device_fallback_is_explicit_and_exact_half_stays_resident() {
    let _guard=SERIAL.lock().unwrap_or_else(|e|e.into_inner());
    let backend=Arc::new(Counting::default());register_backend(DEV,backend.clone());
    let x=Tensor::from_vec(vec![0.25,4.,9.],&[3]).unwrap().to_device(DEV).unwrap().requires_grad_(true).unwrap();
    let cpu_seed=Tensor::full(&[3],0.7);
    let seed=Tensor::full_on(&[3],0.7,DEV).unwrap();
    let (up,down)=(backend.uploads.load(Ordering::SeqCst),backend.downloads.load(Ordering::SeqCst));
    let output=x.pow_scalar(1.7).unwrap();assert_eq!(output.device(),Device::Cpu);
    assert_eq!(backend.uploads.load(Ordering::SeqCst),up);assert_eq!(backend.downloads.load(Ordering::SeqCst)-down,2);
    // The existing functional boundary rejects cross-device requests before
    // dispatch; ordinary backward transfers the final contribution to its leaf.
    for graph in [false,true] {assert!(output.vjp_wrt(&[&x],&cpu_seed,graph).is_err());}
    assert_eq!(backend.uploads.load(Ordering::SeqCst),up);assert_eq!(backend.downloads.load(Ordering::SeqCst)-down,2);
    output.backward_with(&cpu_seed);let gradient=x.grad().unwrap();assert_eq!(gradient.device(),DEV);assert!(!gradient.requires_grad());
    assert_eq!(backend.uploads.load(Ordering::SeqCst)-up,1);assert_eq!(backend.downloads.load(Ordering::SeqCst)-down,3);
    assert_eq!(backend.binaries.load(Ordering::SeqCst),0);x.zero_grad();
    let (up,down)=(backend.uploads.load(Ordering::SeqCst),backend.downloads.load(Ordering::SeqCst));
    let output=x.pow_scalar(0.5).unwrap();assert_eq!(output.device(),DEV);output.vjp_wrt(&[&x],&seed,false).unwrap();
    assert_eq!(backend.uploads.load(Ordering::SeqCst),up);assert_eq!(backend.downloads.load(Ordering::SeqCst),down);
    assert!(x.grad().is_none());
    // An exactly representable forward may need an original-f64 exponent
    // after subtraction in its derivative. Keep that fallback explicit too.
    let output=x.pow_scalar(0.2f32 as f64).unwrap();assert_eq!(output.device(),DEV);
    assert!(output.vjp_wrt(&[&x],&seed,true).is_err());
    assert_eq!(backend.uploads.load(Ordering::SeqCst),up);assert_eq!(backend.downloads.load(Ordering::SeqCst),down);
    let gradient=output.vjp_wrt(&[&x],&seed,false).unwrap().remove(0);assert_eq!(gradient.device(),DEV);
    assert_eq!(backend.uploads.load(Ordering::SeqCst)-up,1);assert_eq!(backend.downloads.load(Ordering::SeqCst)-down,2);
    let (up,down)=(backend.uploads.load(Ordering::SeqCst),backend.downloads.load(Ordering::SeqCst));
    let plain=x.detach();
    assert!(ferro_core::capture::capture(||plain.pow_scalar(1.7)).is_err());
    assert_eq!(backend.uploads.load(Ordering::SeqCst),up);assert_eq!(backend.downloads.load(Ordering::SeqCst),down);
    let captured=ferro_core::capture::capture(||plain.pow_scalar(0.5)).unwrap();
    let chain=ferro_core::graph::CompiledChain::compile(&captured).unwrap();
    assert!(matches!(chain.prepare_static(),Err(ferro_core::Error::Unsupported{op:"counted_static_power",..})));
}
