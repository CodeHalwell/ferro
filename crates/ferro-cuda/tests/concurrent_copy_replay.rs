use ferro_core::dispatch::register_backend;
use ferro_core::graph::CompiledChain;
use ferro_core::{capture, Device, Tensor};
use ferro_cuda::CudaBackend;
use std::process::Command;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

#[test]
fn device_copy_between_replay_leaves_completes() {
    if std::env::var_os("FERRO_CUDA_COPY_CHILD").is_some() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required CUDA: {e}"),
            Err(_) => return,
        };
        register_backend(Device::Cuda(0), backend);
        let a = Tensor::full(&[1024], 2.).to_device(Device::Cuda(0)).unwrap();
        let b = Tensor::full(&[1024], 2.).to_device(Device::Cuda(0)).unwrap();
        let root = capture(|| a.mul(&b).unwrap().add(&a).unwrap());
        let mut graph = CompiledChain::compile(&root).unwrap().prepare_static().unwrap();
        let start = Arc::new(Barrier::new(2));
        let worker = {
            let (a,b,start) = (a.clone(),b.clone(),start.clone());
            std::thread::spawn(move || {
                start.wait();
                for i in 0..500 {
                    // Both directions guarantee both address orders, regardless of allocator order.
                    if i%2 == 0 { a.copy_from(&b).unwrap(); } else { b.copy_from(&a).unwrap(); }
                }
            })
        };
        start.wait();
        for _ in 0..500 {
            graph.replay().unwrap();
            assert_eq!(graph.snapshot().unwrap().to_vec(), vec![6.;1024]);
        }
        worker.join().unwrap();
        a.copy_from(&a).unwrap();
        assert!(a.copy_from(&Tensor::full(&[1], 0.)).is_err());
        assert!(a.copy_from(&Tensor::from_vec_i64(vec![1;1024], &[1024]).unwrap()).is_err());
        let alias = a.detach_copy();
        assert!(a.copy_from(&b).is_err());
        drop(alias);
        a.copy_from(&Tensor::full(&[1024], 3.)).unwrap();
        graph.replay().unwrap();
        assert_eq!(graph.snapshot().unwrap().to_vec(), vec![9.;1024]);
        b.copy_from(&a).unwrap();
        graph.replay().unwrap();
        assert_eq!(graph.snapshot().unwrap().to_vec(), vec![12.;1024]);
        return;
    }
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "device_copy_between_replay_leaves_completes", "--nocapture"])
        .env("FERRO_CUDA_COPY_CHILD", "1").spawn().unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "CUDA concurrent copy child: {status}");
            break;
        }
        if start.elapsed() > Duration::from_secs(90) {
            child.kill().unwrap(); child.wait().unwrap();
            panic!("CUDA concurrent device-copy/prepared-replay child timed out");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
