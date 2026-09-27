//! Non-additive isolated graph GEMM comparison; not a full-model speed claim.
use cudarc::{cublas::{CudaBlas, result as blas, sys::cublasOperation_t::CUBLAS_OP_N as N}, driver::{CudaContext, DevicePtr, DevicePtrMut, sys}};
use std::time::Instant;

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let handle = CudaBlas::new(stream.clone()).unwrap();
    let mut workspace = stream.alloc_zeros::<u8>(4<<20).unwrap();
    let (wp,wg) = workspace.device_ptr_mut(&stream);
    unsafe { cudarc::cublas::sys::cublasSetWorkspace_v2(*handle.handle(),wp as _,4<<20).result().unwrap(); }
    drop(wg);
    for (m,k,n) in [(128,256,256),(128,256,1024),(128,1024,256)] {
        let a = stream.clone_htod(&vec![0.01f32;(m*k) as usize]).unwrap();
        let b = stream.clone_htod(&vec![0.02f32;(k*n) as usize]).unwrap();
        let mut out = stream.alloc_zeros::<f32>((m*n) as usize).unwrap();
        let (ap,ag) = a.device_ptr(&stream);
        let (bp,bg) = b.device_ptr(&stream);
        let (op,og) = out.device_ptr_mut(&stream);
        drop(og); // All raw-pointer work and readback below is on this stream and fenced.
        for round in 0..2 {
            for batched in if round==0 {[false,true]} else {[true,false]} {
                let enqueue = || unsafe {
                    if batched {
                        blas::sgemm_strided_batched(*handle.handle(),N,N,n,m,k,&1.0,bp as _,n,(k*n) as i64,ap as _,k,(m*k) as i64,&0.0,op as _,n,(m*n) as i64,1)
                    } else {
                        blas::sgemm(*handle.handle(),N,N,n,m,k,&1.0,bp as _,n,ap as _,k,&0.0,op as _,n)
                    }.unwrap();
                };
                enqueue();
                stream.synchronize().unwrap();
                stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL).unwrap();
                enqueue();
                let graph = stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH).unwrap().unwrap();
                for _ in 0..10 { graph.launch().unwrap(); }
                stream.synchronize().unwrap();
                let mut samples = Vec::new();
                for _ in 0..40 {
                    let start = Instant::now();
                    graph.launch().unwrap();
                    stream.synchronize().unwrap();
                    samples.push(start.elapsed().as_secs_f64()*1e6);
                }
                let values = stream.clone_dtoh(&out).unwrap();
                assert!(values.iter().all(|v|(*v-k as f32*0.0002).abs()<2e-5));
                let mut ordered = samples.clone();
                ordered.sort_by(f64::total_cmp);
                println!("{{\"m\":{m},\"k\":{k},\"n\":{n},\"round\":{round},\"batched\":{batched},\"median_us\":{},\"samples_us\":{samples:?}}}",(ordered[19]+ordered[20])/2.0);
            }
        }
        drop((ag,bg));
    }
    stream.synchronize().unwrap();
}
