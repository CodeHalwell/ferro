use std::any::Any;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use ferro_core::{capture, Device, Tensor, Result};
use ferro_core::dispatch::{Backend, DeviceBuffer, UnaryKind, BinaryKind, CpuBackend, register_backend};
use ferro_core::graph::CompiledChain;

const DEV: Device = Device::Cuda(29);
static LOCK: Mutex<()> = Mutex::new(());
static COPIES: AtomicUsize = AtomicUsize::new(0);
struct Buf(Mutex<Vec<f32>>);
impl DeviceBuffer for Buf {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.lock().unwrap().len() }
    fn as_any(&self) -> &dyn Any { self }
}
fn data(b: &dyn DeviceBuffer) -> Vec<f32> { b.as_any().downcast_ref::<Buf>().unwrap().0.lock().unwrap().clone() }
fn buf(v: Vec<f32>) -> Box<dyn DeviceBuffer> { Box::new(Buf(Mutex::new(v))) }
struct Mock;
impl Backend for Mock {
    fn unary(&self,k:UnaryKind,x:&[f32])->Vec<f32>{CpuBackend.unary(k,x)}
    fn binary(&self,k:BinaryKind,a:&[f32],b:&[f32])->Vec<f32>{CpuBackend.binary(k,a,b)}
    fn matmul(&self,a:&[f32],b:&[f32],m:usize,k:usize,n:usize)->Vec<f32>{CpuBackend.matmul(a,b,m,k,n)}
    fn alloc_from_host(&self,x:&[f32])->Result<Box<dyn DeviceBuffer>>{Ok(buf(x.to_vec()))}
    fn copy_to_host(&self,x:&dyn DeviceBuffer)->Result<Vec<f32>>{Ok(data(x))}
    fn copy_dev(&self,x:&dyn DeviceBuffer)->Result<Box<dyn DeviceBuffer>>{COPIES.fetch_add(1,Ordering::SeqCst);Ok(buf(data(x)))}
    fn copy_into_dev(&self,dst:&dyn DeviceBuffer,src:&dyn DeviceBuffer)->Result<()>{
        let v=data(src); *dst.as_any().downcast_ref::<Buf>().unwrap().0.lock().unwrap()=v; Ok(())
    }
    fn unary_dev(&self,k:UnaryKind,x:&dyn DeviceBuffer)->Result<Box<dyn DeviceBuffer>>{Ok(buf(CpuBackend.unary(k,&data(x))))}
    fn materialize_dev(&self,x:&dyn DeviceBuffer,shape:&[usize],strides:&[usize],offset:usize)->Result<Box<dyn DeviceBuffer>>{
        let x=data(x);
        Ok(buf((0..shape.iter().product()).map(|mut i| {
            let mut at=offset;
            for d in (0..shape.len()).rev(){at+=(i%shape[d])*strides[d]; i/=shape[d];} x[at]
        }).collect()))
    }
}
fn tensor(v: Vec<f32>) -> Tensor { Tensor::from_vec(v,&[2,3]).unwrap().to_device(DEV).unwrap() }

#[test]
fn ordinary_device_aliases_still_block_writes_with_capture() {
    let _lock=LOCK.lock().unwrap_or_else(|p|p.into_inner());
    register_backend(DEV,Arc::new(Mock));
    for case in 0..3 {
        let x=tensor(vec![1.,2.,3.,4.,5.,6.]);
        let alias=match case {
            0=>x.transpose(0,1).unwrap(),
            1=>x.reshape(&[3,2]).unwrap(),
            _=>x.detach_copy(),
        };
        let before=alias.to_vec();
        let root=capture(||x.transpose(0,1).unwrap().neg());
        let graph=CompiledChain::compile(&root).unwrap();
        let update=tensor(vec![7.,8.,9.,10.,11.,12.]);
        assert!(x.copy_from(&update).is_err());
        assert_eq!(alias.to_vec(),before);
        drop(alias);
        x.copy_from(&update).unwrap();
        assert_eq!(graph.replay().unwrap().to_vec(),vec![-7.,-10.,-8.,-11.,-9.,-12.]);
    }
}

#[test]
fn captured_layout_autograd_and_host_view_semantics_are_unchanged() {
    for transpose in [true,false] {
        let x=Tensor::from_vec(vec![0.2,-0.4,0.7,0.3,-0.8,0.9],&[2,3]).unwrap();
        ferro_core::testkit::grad_check(&[x], |xs| capture(|| {
            let y=if transpose {xs[0].transpose(0,1).unwrap()} else {xs[0].reshape(&[3,2]).unwrap()};
            y.mul(&y).unwrap().sum()
        }));
    }
    let x=Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.],&[2,3]).unwrap();
    let view=capture(||x.transpose(0,1).unwrap());
    x.copy_from(&Tensor::full(&[2,3],7.)).unwrap();
    assert_eq!(view.to_vec(),vec![7.;6]);
}

#[test]
fn retained_captured_reshape_allows_current_input_update() {
    let _lock=LOCK.lock().unwrap_or_else(|p|p.into_inner());
    register_backend(DEV,Arc::new(Mock));
    let x=tensor(vec![1.,2.,3.,4.,5.,6.]);
    let root=capture(||x.reshape(&[3,2]).unwrap().neg());
    let graph=CompiledChain::compile(&root).unwrap();
    x.copy_from(&tensor(vec![7.,8.,9.,10.,11.,12.])).unwrap();
    assert_eq!(graph.replay().unwrap().to_vec(),vec![-7.,-8.,-9.,-10.,-11.,-12.]);
    assert_eq!(root.to_vec(),vec![-1.,-2.,-3.,-4.,-5.,-6.]);
}

#[test]
fn retained_captured_transpose_allows_current_input_update() {
    let _lock=LOCK.lock().unwrap_or_else(|p|p.into_inner());
    register_backend(DEV,Arc::new(Mock));
    let x=tensor(vec![1.,2.,3.,4.,5.,6.]);
    let identity=(x.id(),x._storage_ptr());
    COPIES.store(0,Ordering::SeqCst);
    let root=capture(||x.transpose(0,1).unwrap().neg());
    assert_eq!(COPIES.load(Ordering::SeqCst),1);
    let graph=CompiledChain::compile(&root).unwrap();
    x.copy_from(&tensor(vec![7.,8.,9.,10.,11.,12.])).unwrap();
    assert_eq!(graph.replay().unwrap().to_vec(),vec![-7.,-10.,-8.,-11.,-9.,-12.]);
    assert_eq!(root.to_vec(),vec![-1.,-4.,-2.,-5.,-3.,-6.]);
    assert_eq!((x.id(),x._storage_ptr()),identity);
    assert_eq!(COPIES.load(Ordering::SeqCst),1,"replay must not copy captured layouts");
}
