use ferro_core::{Tensor, Device, Result, Error, capture};
use ferro_core::dispatch::{Backend, DeviceBuffer, CpuBackend, UnaryKind, BinaryKind, register_backend};
use ferro_core::graph::CompiledChain;
use std::any::Any;
use std::sync::{Arc, Mutex};

const DEV: Device = Device::Cuda(43);
static LOCK: Mutex<()> = Mutex::new(());
struct Buf(Vec<f32>);
impl DeviceBuffer for Buf {
    fn device(&self) -> Device { DEV }
    fn len(&self) -> usize { self.0.len() }
    fn as_any(&self) -> &dyn Any { self }
}
fn values(b: &dyn DeviceBuffer) -> &[f32] { &b.as_any().downcast_ref::<Buf>().unwrap().0 }
struct Partial(bool);
impl Backend for Partial {
    fn unary(&self, k: UnaryKind, x: &[f32]) -> Vec<f32> { CpuBackend.unary(k,x) }
    fn binary(&self, k: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> { CpuBackend.binary(k,a,b) }
    fn matmul(&self,a:&[f32],b:&[f32],m:usize,k:usize,n:usize)->Vec<f32>{CpuBackend.matmul(a,b,m,k,n)}
    fn alloc_from_host(&self,x:&[f32])->Result<Box<dyn DeviceBuffer>>{Ok(Box::new(Buf(x.to_vec())))}
    fn copy_to_host(&self,x:&dyn DeviceBuffer)->Result<Vec<f32>>{Ok(values(x).to_vec())}
    fn layer_norm_dev(&self, x: &dyn DeviceBuffer, w: Option<&dyn DeviceBuffer>, b: Option<&dyn DeviceBuffer>, rows: usize, cols: usize, eps: f32)
        -> Result<(Box<dyn DeviceBuffer>, Box<dyn DeviceBuffer>, Box<dyn DeviceBuffer>)> {
        if !self.0 { return Err(Error::Unsupported { op: "layer_norm_dev", msg: "deliberate decline".into() }); }
        let x = Tensor::from_vec(values(x).to_vec(), &[rows,cols])?;
        let w = w.map(|v| Tensor::from_vec(values(v).to_vec(), &[cols]).unwrap());
        let b = b.map(|v| Tensor::from_vec(values(v).to_vec(), &[cols]).unwrap());
        let y = x.layer_norm(w.as_ref(), b.as_ref(), eps)?.to_vec();
        let h = x.layer_norm(None,None,eps)?.to_vec();
        let stds = x.to_vec().chunks(cols).map(|row| {
            let m = row.iter().sum::<f32>()/cols as f32;
            (row.iter().map(|v| (v-m)*(v-m)).sum::<f32>()/cols as f32+eps).sqrt()
        }).collect();
        Ok((Box::new(Buf(y)),Box::new(Buf(h)),Box::new(Buf(stds))))
    }
}
#[test]
fn declined_fusion_and_partial_fused_backward_preserve_values_and_devices() {
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for fused in [false,true] {
        register_backend(DEV,Arc::new(Partial(fused)));
        for mask in 0..4 {
            let run = |device| {
                let x = Tensor::from_vec(vec![0.2,-0.7,1.3,0.5,0.1,-0.9], &[2,3]).unwrap().to_device(device).unwrap().requires_grad_(true).unwrap();
                let w = Tensor::full_on(&[3],1.2,device).unwrap().requires_grad_(true).unwrap();
                let b = Tensor::full_on(&[3],0.3,device).unwrap().requires_grad_(true).unwrap();
                let y = capture(|| x.layer_norm((mask&1!=0).then_some(&w),(mask&2!=0).then_some(&b),1e-5).unwrap());
                if device == DEV { assert_eq!(y.device(), if fused { DEV } else { Device::Cpu }); }
                let replay = CompiledChain::compile(&y).unwrap().replay().unwrap();
                assert!(!replay.requires_grad());
                assert_eq!(replay.to_vec(), y.to_vec());
                let seed = Tensor::from_vec(vec![0.1,0.3,-0.2,0.8,-0.4,0.2], &[2,3]).unwrap().to_device(y.device()).unwrap();
                y.backward_with(&seed);
                let mut result = y.to_vec();
                for t in [Some(&x),(mask&1!=0).then_some(&w),(mask&2!=0).then_some(&b)].into_iter().flatten() {
                    let grad = t.grad().unwrap();
                    assert_eq!(grad.device(),device);
                    result.extend(grad.to_vec());
                }
                result
            };
            for (a,b) in run(DEV).iter().zip(run(Device::Cpu)) { assert!((a-b).abs()<1e-5, "{a} != {b}"); }
        }
    }
}
