use super::*;
use crate::dispatch::{Backend, register_backend};
use std::any::Any;
use std::sync::Weak;

const DEV: Device = Device::Cuda(30);
struct Buf(Vec<f32>);
struct Idx(Vec<i64>);
macro_rules! buffer { ($t:ty) => { impl DeviceBuffer for $t {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}}; }
buffer!(Buf); buffer!(Idx);
struct GatherOnly(Weak<StorageCell>);
impl Backend for GatherOnly {
    fn unary(&self,_:UnaryKind,_:&[f32])->Vec<f32>{unreachable!()}
    fn binary(&self,_:BinaryKind,_:&[f32],_:&[f32])->Vec<f32>{unreachable!()}
    fn matmul(&self,_:&[f32],_:&[f32],_:usize,_:usize,_:usize)->Vec<f32>{unreachable!()}
    fn alloc_i64_from_host(&self,x:&[i64])->Result<Box<dyn DeviceBuffer>>{
        let cell=self.0.upgrade().unwrap();
        assert!(cell.data.try_write().is_ok(), "layout index upload must not retain the source lock");
        Ok(Box::new(Idx(x.to_vec())))
    }
    fn gather_rows_dev(&self,x:&dyn DeviceBuffer,i:&dyn DeviceBuffer,_:usize,_:usize)->Result<Box<dyn DeviceBuffer>>{
        let cell=self.0.upgrade().unwrap();
        assert!(cell.data.try_write().is_err(), "gather must reacquire the source lock");
        let x=&x.as_any().downcast_ref::<Buf>().unwrap().0;
        let i=&i.as_any().downcast_ref::<Idx>().unwrap().0;
        Ok(Box::new(Buf(i.iter().map(|&i|x[i as usize]).collect())))
    }
    fn copy_to_host(&self,x:&dyn DeviceBuffer)->Result<Vec<f32>>{
        Ok(x.as_any().downcast_ref::<Buf>().unwrap().0.clone())
    }
}
#[test]
fn gather_fallback_unlocks_during_index_upload() {
    let x=device_leaf(Box::new(Buf(vec![1.,2.,3.,4.,5.,6.])), &[2,3], DEV);
    register_backend(DEV, Arc::new(GatherOnly(Arc::downgrade(&x.0.storage))));
    let y=x.transpose(0,1).unwrap().reshape(&[6]).unwrap();
    assert_eq!(y.device(),DEV);
    assert_eq!(y.to_vec(),vec![1.,4.,2.,5.,3.,6.]);
}
