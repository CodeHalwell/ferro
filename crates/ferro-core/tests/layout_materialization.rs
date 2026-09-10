use std::any::Any;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use ferro_core::{Tensor, Device, Result};
use ferro_core::dispatch::{Backend, DeviceBuffer, UnaryKind, BinaryKind, CpuBackend, register_backend};
const DEV: Device = Device::Cuda(27);
static INDEX_UPLOADS: AtomicUsize = AtomicUsize::new(0);
static DOWNLOADS: AtomicUsize = AtomicUsize::new(0);
struct Buf(Vec<f32>);
struct Idx(Vec<i64>);
macro_rules! buffer { ($t:ty) => { impl DeviceBuffer for $t {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}}; }
buffer!(Buf); buffer!(Idx);
fn data(b: &dyn DeviceBuffer) -> &[f32] { &b.as_any().downcast_ref::<Buf>().unwrap().0 }
struct Mock;
impl Backend for Mock {
    fn materialize_dev(&self, x: &dyn DeviceBuffer, shape: &[usize], strides: &[usize], offset: usize) -> Result<Box<dyn DeviceBuffer>> {
        let values=(0..shape.iter().product()).map(|mut i| {
            let mut at=offset;
            for d in (0..shape.len()).rev() { at+=(i%shape[d])*strides[d]; i/=shape[d]; }
            data(x)[at]
        }).collect();
        Ok(Box::new(Buf(values)))
    }
    fn unary(&self, k: UnaryKind, x: &[f32]) -> Vec<f32> { CpuBackend.unary(k,x) }
    fn binary(&self, k: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> { CpuBackend.binary(k,a,b) }
    fn matmul(&self,a:&[f32],b:&[f32],m:usize,k:usize,n:usize)->Vec<f32>{CpuBackend.matmul(a,b,m,k,n)}
    fn alloc_from_host(&self,x:&[f32])->Result<Box<dyn DeviceBuffer>>{Ok(Box::new(Buf(x.to_vec())))}
    fn copy_to_host(&self,x:&dyn DeviceBuffer)->Result<Vec<f32>>{DOWNLOADS.fetch_add(1,Ordering::SeqCst);Ok(data(x).to_vec())}
    fn fill_dev(&self,v:f32,n:usize)->Result<Box<dyn DeviceBuffer>>{Ok(Box::new(Buf(vec![v;n])))}
    fn unary_dev(&self,k:UnaryKind,x:&dyn DeviceBuffer)->Result<Box<dyn DeviceBuffer>>{Ok(Box::new(Buf(CpuBackend.unary(k,data(x)))))}
    fn binary_dev(&self,k:BinaryKind,a:&dyn DeviceBuffer,b:&dyn DeviceBuffer)->Result<Box<dyn DeviceBuffer>>{Ok(Box::new(Buf(CpuBackend.binary(k,data(a),data(b)))))}
    fn binary_bc_dev(&self,k:BinaryKind,a:&dyn DeviceBuffer,sa:&[usize],b:&dyn DeviceBuffer,sb:&[usize],_:&[usize])->Result<Box<dyn DeviceBuffer>>{
        let a=Tensor::from_vec(data(a).to_vec(),sa)?;let b=Tensor::from_vec(data(b).to_vec(),sb)?;
        let y=match k{BinaryKind::Add=>a.add(&b),BinaryKind::Sub=>a.sub(&b),BinaryKind::Mul=>a.mul(&b),BinaryKind::Div=>a.div(&b)}?;
        Ok(Box::new(Buf(y.to_vec())))
    }
    fn sum_dim_dev(&self,x:&dyn DeviceBuffer,s:&[usize],d:usize)->Result<Box<dyn DeviceBuffer>>{Ok(Box::new(Buf(Tensor::from_vec(data(x).to_vec(),s)?.sum_dim(d,true)?.to_vec())))}
    fn alloc_i64_from_host(&self,x:&[i64])->Result<Box<dyn DeviceBuffer>>{INDEX_UPLOADS.fetch_add(1,Ordering::SeqCst);Ok(Box::new(Idx(x.to_vec())))}
    fn gather_rows_dev(&self,x:&dyn DeviceBuffer,i:&dyn DeviceBuffer,_:usize,inner:usize)->Result<Box<dyn DeviceBuffer>>{
        let idx=&i.as_any().downcast_ref::<Idx>().unwrap().0;
        Ok(Box::new(Buf(idx.iter().flat_map(|&j|data(x)[j as usize*inner..(j as usize+1)*inner].iter().copied()).collect())))
    }
}

#[test]
fn layout_materialization_uses_no_index_uploads() {
    register_backend(DEV,Arc::new(Mock));
    let x=Tensor::from_vec((0..24).map(|i|i as f32).collect(), &[2,3,4]).unwrap().to_device(DEV).unwrap();
    let y=x.transpose(0,1).unwrap().reshape(&[3,8]).unwrap();
    assert_eq!(INDEX_UPLOADS.load(Ordering::SeqCst),0);
    assert_eq!(DOWNLOADS.load(Ordering::SeqCst),0);
    assert_eq!(y.to_vec(),vec![0.,1.,2.,3.,12.,13.,14.,15.,4.,5.,6.,7.,16.,17.,18.,19.,8.,9.,10.,11.,20.,21.,22.,23.]);
}
