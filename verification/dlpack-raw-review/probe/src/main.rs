use cudarc::driver::{result, sys, CudaContext, DevicePtr};

fn main() {
    // Independent of Ferro, Python, capsules, and import_cuda.
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let expected = [0f32, 1., 2., 3., 4., 5.];
    let mut flags = 0;
    unsafe { sys::cuStreamGetFlags(stream.cu_stream(), &mut flags).result().unwrap(); }
    assert_eq!(flags, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32);
    let data = stream.clone_htod(&expected).unwrap();
    let (ptr, usage) = data.device_ptr(&stream);
    let mut before = [0f32; 6];
    unsafe { result::memcpy_dtoh_sync(&mut before, ptr).unwrap(); }
    stream.synchronize().unwrap();
    let mut after = [0f32; 6];
    unsafe { result::memcpy_dtoh_sync(&mut after, ptr).unwrap(); }
    drop(usage);
    ctx.check_err().unwrap();
    println!("nonblocking_flags={flags}; unfenced={before:?}; fenced={after:?}");
    // Unfenced output is a race, not a contract; only the fenced value is asserted.
    assert_eq!(after, expected);
}
