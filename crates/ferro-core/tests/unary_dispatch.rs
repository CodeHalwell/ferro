use std::sync::{Arc, Mutex, atomic::{AtomicUsize,Ordering}};
use ferro_core::{DType, Device, Tensor};
use ferro_core::dispatch::{register_backend,Backend,BinaryKind,CpuBackend,UnaryKind};

static SERIAL:Mutex<()>=Mutex::new(());
#[derive(Default)]
struct Counting { calls:AtomicUsize, seen:Mutex<Vec<f32>> }
impl Backend for Counting {
    fn unary(&self,kind:UnaryKind,x:&[f32])->Vec<f32>{
        self.calls.fetch_add(1,Ordering::SeqCst);*self.seen.lock().unwrap_or_else(|e|e.into_inner())=x.to_vec();CpuBackend.unary(kind,x)
    }
    fn binary(&self,kind:BinaryKind,a:&[f32],b:&[f32])->Vec<f32>{CpuBackend.binary(kind,a,b)}
    fn matmul(&self,a:&[f32],b:&[f32],m:usize,k:usize,n:usize)->Vec<f32>{CpuBackend.matmul(a,b,m,k,n)}
}

#[test]
fn physical_unary_iteration_retains_the_registered_kernel_and_rejects_capacity_before_dispatch() {
    let _guard=SERIAL.lock().unwrap_or_else(|e|e.into_inner());
    let backend=Arc::new(Counting::default());register_backend(Device::Cpu,backend.clone());
    let x=Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.],&[2,3]).unwrap();
    let output=x.transpose(0,1).unwrap().exp();
    assert_eq!(output.strides(),&[1,3]);assert_eq!(backend.calls.load(Ordering::SeqCst),1);
    assert_eq!(*backend.seen.lock().unwrap_or_else(|e|e.into_inner()),vec![1.,2.,3.,4.,5.,6.]);
    x.narrow(0,1,1).unwrap().neg();
    assert_eq!(backend.calls.load(Ordering::SeqCst),2);
    assert_eq!(*backend.seen.lock().unwrap_or_else(|e|e.into_inner()),vec![4.,5.,6.]);
    let huge=Tensor::scalar(1.).expand(&[isize::MAX as usize/4+1]).unwrap();
    assert!(huge.sign().is_err());assert_eq!(backend.calls.load(Ordering::SeqCst),2);
    x.to_dtype(DType::F64).transpose(0,1).unwrap().exp();assert_eq!(backend.calls.load(Ordering::SeqCst),2);
    let leaf=x.transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
    let y=leaf.exp();assert_eq!(backend.calls.load(Ordering::SeqCst),3);
    y.sum().backward();assert_eq!(leaf.grad().unwrap().strides(),&[1,3]);
    assert_eq!(backend.calls.load(Ordering::SeqCst),3);
}

#[test]
fn half_power_forward_and_cpu_adjoint_keep_registered_unary_dispatch() {
    let _guard=SERIAL.lock().unwrap_or_else(|e|e.into_inner());
    let backend=Arc::new(Counting::default());register_backend(Device::Cpu,backend.clone());
    let values=vec![0.25,0.5,4.,9.];
    let x=Tensor::from_vec(values.clone(),&[2,2]).unwrap().transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
    let y=x.powf(0.5);assert_eq!(backend.calls.load(Ordering::SeqCst),1);assert_eq!(y.strides(),&[1,2]);
    assert_eq!(*backend.seen.lock().unwrap_or_else(|e|e.into_inner()),values);
    let seed=Tensor::scalar(1.).expand(x.shape()).unwrap();
    let g=Tensor::vjp_many_wrt_optional(&[&y],&[&x],&[&seed],false).unwrap().remove(0).unwrap();
    assert_eq!(backend.calls.load(Ordering::SeqCst),2);assert_eq!(g.strides(),&[1,2]);
    assert_eq!(*backend.seen.lock().unwrap_or_else(|e|e.into_inner()),values);
    assert_eq!(g.to_vec(),x.to_vec().into_iter().map(|x|0.5*x.sqrt().recip()).collect::<Vec<_>>());assert!(x.grad().is_none());
}
