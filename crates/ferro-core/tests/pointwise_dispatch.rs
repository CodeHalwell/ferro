use std::sync::{Arc, Mutex, atomic::{AtomicUsize,Ordering}};
use ferro_core::{Device,Tensor};
use ferro_core::dispatch::{register_backend,Backend,BinaryKind,CpuBackend,UnaryKind};

static SERIAL:Mutex<()>=Mutex::new(());
#[derive(Default)]
struct Counting {binary:AtomicUsize,broadcast:AtomicUsize,seen:Mutex<Vec<f32>>}
impl Backend for Counting {
    fn unary(&self,kind:UnaryKind,x:&[f32])->Vec<f32>{CpuBackend.unary(kind,x)}
    fn binary(&self,kind:BinaryKind,a:&[f32],b:&[f32])->Vec<f32>{
        self.binary.fetch_add(1,Ordering::SeqCst);*self.seen.lock().unwrap()=a.to_vec();CpuBackend.binary(kind,a,b)
    }
    fn binary_bc(&self,kind:BinaryKind,a:&[f32],sa:&[usize],b:&[f32],sb:&[usize],out:&[usize])->Vec<f32>{
        self.broadcast.fetch_add(1,Ordering::SeqCst);CpuBackend.binary_bc(kind,a,sa,b,sb,out)
    }
    fn matmul(&self,a:&[f32],b:&[f32],m:usize,k:usize,n:usize)->Vec<f32>{CpuBackend.matmul(a,b,m,k,n)}
}

#[test]
fn physical_iteration_retains_registered_cpu_binary_and_contiguous_broadcast_dispatch() {
    let _guard=SERIAL.lock().unwrap_or_else(|e|e.into_inner());
    let backend=Arc::new(Counting::default());register_backend(Device::Cpu,backend.clone());
    let x=Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.],&[2,3]).unwrap();
    let out=x.mul(&x).unwrap();assert_eq!(out.to_vec(),vec![1.,4.,9.,16.,25.,36.]);
    assert_eq!(backend.binary.load(Ordering::SeqCst),1);
    x.add(&Tensor::ones(&[3])).unwrap();assert_eq!(backend.broadcast.load(Ordering::SeqCst),1);
    let view=x.transpose(0,1).unwrap();let out=view.mul(&view).unwrap();
    assert_eq!(out.strides(),&[1,3]);assert_eq!(out.to_vec(),vec![1.,16.,4.,25.,9.,36.]);
    assert_eq!(backend.binary.load(Ordering::SeqCst),2);
    assert_eq!(*backend.seen.lock().unwrap(),vec![1.,2.,3.,4.,5.,6.]);
    let leaf=view.detach().requires_grad_(true).unwrap();leaf.mul(&leaf).unwrap().sum().backward();
    assert_eq!(leaf.grad().unwrap().to_vec(),vec![2.,8.,4.,10.,6.,12.]);
    assert!(backend.binary.load(Ordering::SeqCst)>=5);
    let count=backend.binary.load(Ordering::SeqCst);
    let typed=view.to_dtype_autograd(ferro_core::DType::F64,false).unwrap();typed.mul(&typed).unwrap();
    assert_eq!(backend.binary.load(Ordering::SeqCst),count);
}
