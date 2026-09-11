use ferro_core::{Tensor, Device, capture};
use ferro_core::graph::CompiledChain;

#[test]
fn fused_layer_norm_kernel_bounds_and_stability() {
    use ferro_core::dispatch::Backend;
    let backend = match ferro_cuda::CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => {
            assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}");
            eprintln!("SKIP CUDA: {e}");
            return;
        }
    };
    let x = backend.alloc_from_host(&[1.,2.,3.]).unwrap();
    assert_eq!(backend.layer_norm_counts(), (0,1,0));
    backend.copy_to_host(x.as_ref()).unwrap();
    assert_eq!(backend.layer_norm_counts(), (0,1,1));
    for (rows,cols) in [(1,0), (2,3), (usize::MAX,2)] {
        assert!(backend.layer_norm_dev(x.as_ref(), None, None, rows,cols,1e-5).is_err());
    }
    assert!(backend.layer_norm_dev(x.as_ref(), Some(x.as_ref()), None,3,1,1e-5).is_err());
    let empty = backend.alloc_from_host(&[]).unwrap();
    let counts = backend.layer_norm_counts();
    let (y,h,s) = backend.layer_norm_dev(empty.as_ref(),None,None,0,3,1e-5).unwrap();
    assert_eq!((y.len(),h.len(),s.len()),(0,0,0));
    assert_eq!(backend.layer_norm_counts(),counts);
    for (cols,offset,step) in [(33,1e8f32,8f32), (4097,1e20,1e14), (7,3.0,0.0)] {
        let data: Vec<f32> = (0..2*cols).map(|i| offset + (i%19) as f32*step).collect();
        let x = backend.alloc_from_host(&data).unwrap();
        let (y,_,_) = backend.layer_norm_dev(x.as_ref(),None,None,2,cols,1e-5).unwrap();
        let actual = backend.copy_to_host(y.as_ref()).unwrap();
        for r in 0..2 {
            let row = &data[r*cols..(r+1)*cols];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>()/cols as f64;
            let var = row.iter().map(|&v| (v as f64-mean).powi(2)).sum::<f64>()/cols as f64;
            for i in 0..cols {
                let want = ((row[i] as f64-mean)/(var+1e-5).sqrt()) as f32;
                assert!((actual[r*cols+i]-want).abs() < 2e-6);
            }
        }
    }
}


#[test]
fn fused_layer_norm_is_one_resident_operation() {
    if let Err(e) = ferro_cuda::install(0) {
        assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}");
        eprintln!("SKIP CUDA: {e}");
        return;
    }
    let backend = ferro_cuda::cuda_backend().unwrap();
    // Same storage may occupy all three argument positions under a writer.
    let alias = Tensor::full_on(&[31], 0.5, Device::Cuda(0)).unwrap();
    let a = alias.clone();
    let (tx,rx) = std::sync::mpsc::channel();
    let writer = std::thread::spawn(move || { for _ in 0..10000 { a.fill_(0.75).unwrap(); } });
    std::thread::spawn(move || {
        for _ in 0..1000 { alias.layer_norm(Some(&alias), Some(&alias), 1e-5).unwrap(); }
        writer.join().unwrap();
        tx.send(()).unwrap();
    });
    rx.recv_timeout(std::time::Duration::from_secs(10)).expect("aliased LayerNorm deadlocked with writer");
    for shape in [vec![7], vec![2,3,7], vec![0,7]] {
        let n = shape.iter().product();
        let cpu = Tensor::from_vec((0..n).map(|i| (i%17) as f32*0.1).collect(), &shape).unwrap();
        let x = cpu.to_device(Device::Cuda(0)).unwrap();
        let allocations = backend.alloc_stats();
        let counts = backend.layer_norm_counts();
        let got = x.layer_norm(None,None,1e-5).unwrap();
        assert_eq!(backend.alloc_stats().since(&allocations).requests, 1);
        let after = backend.layer_norm_counts();
        assert_eq!((after.0-counts.0, after.1-counts.1, after.2-counts.2), (usize::from(n != 0),0,0));
        assert_eq!(got.shape(),shape);
        for (a,b) in got.to_vec().iter().zip(cpu.layer_norm(None,None,1e-5).unwrap().to_vec()) { assert!((a-b).abs()<1e-5); }
    }
    for training in [false, true] {
    for d in [1, 3, 31, 256, 513, 8193] {
        for affine in 0..4 {
            let cpu = Tensor::from_vec((0..3*d).map(|i| 1000.0 + (i % 17) as f32 * 0.125).collect(), &[3,d]).unwrap();
            let x = cpu.to_device(Device::Cuda(0)).unwrap().requires_grad_(training).unwrap();
            let w = Tensor::full(&[d], 1.25).to_device(Device::Cuda(0)).unwrap().requires_grad_(training).unwrap();
            let b = Tensor::full(&[d], -0.3).to_device(Device::Cuda(0)).unwrap().requires_grad_(training).unwrap();
            let weight = (affine & 1 != 0).then_some(&w);
            let bias = (affine & 2 != 0).then_some(&b);
            let transfers = backend.layer_norm_counts();
            let before = backend.pointwise_launch_counts();
            let allocations = backend.alloc_stats();
            let root = capture(|| x.layer_norm(weight, bias, 1e-5).unwrap());
            assert_eq!(backend.alloc_stats().since(&allocations).requests, if training { 3 } else { 1 }, "inference must allocate only its output");
            assert_eq!(backend.pointwise_launch_counts(), before, "LayerNorm must not enqueue decomposed pointwise kernels");
            let after = backend.layer_norm_counts();
            assert_eq!((after.0-transfers.0, after.1-transfers.1, after.2-transfers.2), (1,0,0));
            let graph = CompiledChain::compile(&root).unwrap();
            assert_eq!(graph.num_runs(), 1);
            let wc = weight.map(|_| Tensor::full(&[d], 1.25));
            let bc = bias.map(|_| Tensor::full(&[d], -0.3));
            let cpu = cpu.requires_grad_(training).unwrap();
            let wc = wc.map(|t| t.requires_grad_(training).unwrap());
            let bc = bc.map(|t| t.requires_grad_(training).unwrap());
            let reference = cpu.layer_norm(wc.as_ref(), bc.as_ref(), 1e-5).unwrap();
            let expected = reference.to_vec();
            let counts = backend.layer_norm_counts();
            let allocations = backend.alloc_stats();
            let replay = capture(|| graph.replay().unwrap());
            assert_eq!(backend.alloc_stats().since(&allocations).requests, 1, "replay must allocate only its output");
            assert!(!replay.requires_grad());
            let after = backend.layer_norm_counts();
            assert_eq!((after.0-counts.0, after.1-counts.1, after.2-counts.2), (1,0,0));
            for got in [root.to_vec(), replay.to_vec()] {
                for (a,b) in got.iter().zip(&expected) { assert!((a-b).abs() < 2e-3, "d={d}: {a} != {b}"); }
            }
            if training {
                if weight.is_some() { assert!(w.fill_(0.0).is_err(), "saved affine snapshot must remain protected"); }
                let seed = Tensor::from_vec((0..3*d).map(|i| (i%11) as f32/11.0).collect(), &[3,d]).unwrap();
                reference.mul(&seed).unwrap().sum().backward();
                root.mul(&seed.to_device(Device::Cuda(0)).unwrap()).unwrap().sum().backward();
                for (a,b) in [(Some(&x), Some(&cpu)), (weight, wc.as_ref()), (bias, bc.as_ref())] {
                    if let (Some(a),Some(b)) = (a,b) {
                        assert_eq!(a.grad().unwrap().device(), Device::Cuda(0));
                        for (a,b) in a.grad().unwrap().to_vec().iter().zip(b.grad().unwrap().to_vec()) {
                            assert!((a-b).abs() < 3e-3, "gradient d={d}: {a} != {b}");
                        }
                    }
                }
            } else {
                let changed = Tensor::from_vec((0..3*d).map(|i| (i%13) as f32*0.13).collect(), &[3,d]).unwrap();
                x.copy_from(&changed.to_device(Device::Cuda(0)).unwrap()).unwrap();
                w.fill_(0.7).unwrap();
                b.fill_(0.4).unwrap();
                let want = changed.layer_norm(weight.map(|_| &w), bias.map(|_| &b), 1e-5).unwrap().to_vec();
                let got = graph.replay().unwrap().to_vec();
                for (a,b) in got.iter().zip(want) { assert!((a-b).abs() < 2e-3); }
            }
        }
    }
    }
}
