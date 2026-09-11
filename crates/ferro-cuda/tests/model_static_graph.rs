use std::sync::Arc;

#[test]
fn malformed_static_plan_is_rejected_without_panicking_or_allocating() {
    use ferro_core::dispatch::{Backend,StaticRun,StaticOp,ChainStepRef,DeviceBuffer,BinaryKind};
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    let x:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[1.,2.]).unwrap());
    let runs=[StaticRun {inputs:vec![0],shape:vec![2],op:StaticOp::Pointwise(vec![ChainStepRef::Binary {kind:BinaryKind::Add,other:usize::MAX}])}];
    let before=b.alloc_stats().requests;
    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||Backend::prepare_static(b.clone(),&runs,vec![x])));
    assert!(result.is_ok(),"malformed public plan must not panic");
    assert!(result.unwrap().is_err());
    assert_eq!(b.alloc_stats().requests,before);
}


#[test]
fn static_replay_orders_concurrent_whole_tensor_updates() {
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    register_backend(Device::Cuda(0),b);
    let x=Tensor::full(&[1024],1.).to_device(Device::Cuda(0)).unwrap();
    let root=capture(||x.mul(&x).unwrap().add(&x).unwrap());
    let mut graph=CompiledChain::compile(&root).unwrap().prepare_static().unwrap();
    let writer=x.clone();
    let worker=std::thread::spawn(move || {
        let one=Tensor::full(&[1024],1.).to_device(Device::Cuda(0)).unwrap();
        let two=Tensor::full(&[1024],2.).to_device(Device::Cuda(0)).unwrap();
        for i in 0..100 { writer.copy_from(if i%2==0 {&one} else {&two}).unwrap(); }
    });
    for _ in 0..100 {
        graph.replay().unwrap();
        let values=graph.snapshot().unwrap().to_vec();
        assert!(values[0]==2. || values[0]==6.);
        assert!(values.iter().all(|v|*v==values[0]));
    }
    worker.join().unwrap();
    drop(x); drop(root);
    graph.replay().unwrap();
    assert_eq!(graph.snapshot().unwrap().to_vec(),vec![6.;1024]);
}


#[test]
fn static_layer_norm_supports_each_affine_combination() {
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    register_backend(Device::Cuda(0),b);
    let x=Tensor::from_vec(vec![-1.,2.,3.,4.],&[2,2]).unwrap().to_device(Device::Cuda(0)).unwrap();
    let w=Tensor::full(&[2],1.1).to_device(Device::Cuda(0)).unwrap();
    let bias=Tensor::full(&[2],0.1).to_device(Device::Cuda(0)).unwrap();
    for hw in [false,true] { for hb in [false,true] {
        let forward=||x.layer_norm(hw.then_some(&w),hb.then_some(&bias),1e-5).unwrap();
        let root=capture(forward);
        let compiled=CompiledChain::compile(&root).unwrap();
        let mut graph=compiled.prepare_static().unwrap();
        graph.replay().unwrap();
        assert_eq!(graph.snapshot().unwrap().to_vec(),forward().to_vec());
    }}
}


#[test]
fn compiled_static_attention_mlp_updates_input_and_weights() {
    let b = match CudaBackend::new(0) {
        Ok(b) => Arc::new(b),
        Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required CUDA: {e}"),
        Err(_) => return,
    };
    register_backend(Device::Cuda(0), b.clone());
    let dev = |t: Tensor| t.to_device(Device::Cuda(0)).unwrap();
    let x = dev(Tensor::from_vec((0..24).map(|i| i as f32 * 0.03 - 0.2).collect(), &[3,8]).unwrap());
    let w = dev(Tensor::full(&[8], 1.1));
    let bias = dev(Tensor::full(&[8], 0.05));
    let projection = dev(Tensor::from_vec((0..64).map(|i| (i as f32 * 0.13).sin()*0.2).collect(), &[8,8]).unwrap());
    let forward = || {
        let z = x.layer_norm(Some(&w), Some(&bias), 1e-5).unwrap();
        let q = z.reshape(&[3,2,4]).unwrap().transpose(0,1).unwrap();
        let scores = q.bmm(&q.transpose(1,2).unwrap()).unwrap();
        let a = scores.softmax(2).unwrap().bmm(&q).unwrap();
        let residual = x.add(&a.transpose(0,1).unwrap().reshape(&[3,8]).unwrap()).unwrap();
        residual.add(&residual.matmul(&projection).unwrap().add(&bias).unwrap().gelu().matmul(&projection).unwrap()).unwrap().sum_dim(0,false).unwrap()
    };
    let root = capture(forward);
    let compiled = CompiledChain::compile(&root).unwrap();
    let mut graph = compiled.prepare_static().unwrap();
    for iteration in 0..3 {
        x.copy_from(&dev(Tensor::from_vec((0..24).map(|i| ((i+iteration) as f32*0.7).sin()).collect(), &[3,8]).unwrap())).unwrap();
        w.copy_from(&dev(Tensor::full(&[8], 0.8+iteration as f32*0.1))).unwrap();
        let allocations=b.alloc_stats().requests;
        let transfers=b.layer_norm_counts();
        let launches=b.pointwise_launch_counts();
        graph.replay().unwrap();
        assert_eq!(b.alloc_stats().requests,allocations);
        assert_eq!(b.layer_norm_counts(),transfers);
        assert_eq!(b.pointwise_launch_counts(),launches);
        let actual = graph.snapshot().unwrap().to_vec();
        let expected = forward().to_vec();
        for (a,e) in actual.iter().zip(expected) { assert!((a-e).abs()<2e-5,"{a} vs {e}"); }
    }
}


#[test]
fn compiled_static_matmul_reads_current_weights() {
    let b = match CudaBackend::new(0) {
        Ok(b) => Arc::new(b),
        Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required CUDA: {e}"),
        Err(_) => return,
    };
    register_backend(Device::Cuda(0), b);
    let x = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], &[2,3]).unwrap().to_device(Device::Cuda(0)).unwrap();
    let w = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], &[3,2]).unwrap().to_device(Device::Cuda(0)).unwrap();
    let root = capture(|| x.matmul(&w).unwrap().relu());
    let compiled = CompiledChain::compile(&root).unwrap();
    let mut graph = compiled.prepare_static().unwrap();
    for value in [1.,2.,-1.] {
        w.copy_from(&Tensor::full(&[3,2], value).to_device(Device::Cuda(0)).unwrap()).unwrap();
        graph.replay().unwrap();
        assert_eq!(graph.snapshot().unwrap().to_vec(), compiled.replay().unwrap().to_vec());
    }
}

use ferro_core::{capture, Device, Tensor};
use ferro_core::dispatch::register_backend;
use ferro_core::graph::CompiledChain;
use ferro_cuda::CudaBackend;

#[test]
fn compiled_static_pointwise_reads_current_leaves_and_retains_snapshots() {
    let b = match CudaBackend::new(0) {
        Ok(b) => Arc::new(b),
        Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some() => panic!("required CUDA: {e}"),
        Err(_) => return,
    };
    register_backend(Device::Cuda(0), b);
    let tensor = |v| Tensor::from_vec(v, &[4]).unwrap().to_device(Device::Cuda(0)).unwrap();
    let x = tensor(vec![-1.,2.,-3.,4.]);
    let w = tensor(vec![2.,3.,4.,5.]);
    let root = capture(|| x.relu().add(&x.mul(&w).unwrap()).unwrap());
    let compiled = CompiledChain::compile(&root).unwrap();
    let mut graph = compiled.prepare_static().unwrap();
    graph.replay().unwrap();
    let old = graph.snapshot().unwrap();
    assert_eq!(old.to_vec(), vec![-2.,8.,-12.,24.]);
    x.copy_from(&tensor(vec![1.,-2.,3.,-4.])).unwrap();
    w.copy_from(&tensor(vec![3.,4.,5.,6.])).unwrap();
    graph.replay().unwrap();
    assert_eq!(graph.snapshot().unwrap().to_vec(), compiled.replay().unwrap().to_vec());
    assert_eq!(old.to_vec(), vec![-2.,8.,-12.,24.]);
    assert_eq!(graph.replay_count(), 2);
}
