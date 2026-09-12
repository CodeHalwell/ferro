use std::sync::{Arc, Mutex};

#[test]
fn static_softmax_arbitrary_axes_read_fresh_inputs() {
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    register_backend(Device::Cuda(0),b.clone());
    let mut high_rank=vec![1;17]; high_rank[0]=2; high_rank[8]=3; high_rank[16]=4;
    for shape in [vec![3],vec![2,3],vec![2,3,4],vec![2,1,3,2],vec![2,257,3],high_rank] {
        for axis in 0..shape.len() {
            let n=shape.iter().product();
            let make=|phase:f32| Tensor::from_vec((0..n).map(|i| (i as f32*0.37+phase).sin()*3.).collect(),&shape).unwrap();
            let x=make(0.).to_device(Device::Cuda(0)).unwrap();
            let root=capture(||x.softmax(axis).unwrap());
            let mut graph=CompiledChain::compile(&root).unwrap().prepare_static().unwrap();
            graph.replay().unwrap();
            let old=graph.snapshot().unwrap();
            let saved=old.to_vec();
            for phase in [0.4,1.2] {
                let cpu=make(phase);
                x.copy_from(&cpu.to_device(Device::Cuda(0)).unwrap()).unwrap();
                let before=b.alloc_stats().requests;
                graph.replay().unwrap();
                assert_eq!(b.alloc_stats().requests,before);
                let expected=cpu.softmax(axis).unwrap().to_vec();
                let got=graph.snapshot().unwrap();
                assert_eq!(got.shape(),shape);
                for (a,e) in got.to_vec().iter().zip(expected) { assert!((a-e).abs()<2e-6,"{a} vs {e}"); }
            }
            assert_eq!(graph.replay_count(),3);
            drop(graph);
            assert_eq!(old.to_vec(),saved);
        }
    }
}


#[test]
fn static_expanding_seed_preserves_operand_order_and_fresh_weights() {
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    register_backend(Device::Cuda(0),b.clone());
    for (sa,sb) in [(vec![],vec![2,3]),(vec![3],vec![2,3]),(vec![2,1],vec![1,3]),(vec![1,2,1],vec![3,1,4])] {
        for first in [true,false] { for divide in [true,false] {
            let make=|s:&[usize],offset:f32| Tensor::from_vec((0..s.iter().product()).map(|i|offset+i as f32*0.2).collect(),s).unwrap();
            let a=make(&sa,2.).to_device(Device::Cuda(0)).unwrap();
            let w=make(&sb,1.).to_device(Device::Cuda(0)).unwrap();
            let forward=|a:&Tensor,w:&Tensor| { let a=if first {a.clone()} else {a.relu()}; if divide {a.div(w).unwrap()} else {a.sub(w).unwrap()} };
            let root=capture(||forward(&a,&w).relu());
            let mut graph=CompiledChain::compile(&root).unwrap().prepare_static().unwrap();
            graph.replay().unwrap();
            let saved=graph.snapshot().unwrap(); let old=saved.to_vec();
            for offset in [3.,4.] {
                let ca=make(&sa,offset); let cw=make(&sb,offset-1.);
                a.copy_from(&ca.to_device(Device::Cuda(0)).unwrap()).unwrap();
                w.copy_from(&cw.to_device(Device::Cuda(0)).unwrap()).unwrap();
                let before=b.alloc_stats().requests;
                graph.replay().unwrap(); assert_eq!(b.alloc_stats().requests,before);
                let got=graph.snapshot().unwrap(); let expected=forward(&ca,&cw).relu();
                assert_eq!(got.shape(),expected.shape());
                for (a,e) in got.to_vec().iter().zip(expected.to_vec()) { assert!((a-e).abs()<1e-6); }
            }
            drop(graph); assert_eq!(saved.to_vec(),old);
        }}
    }
}

#[test]
fn static_empty_shapes_replay_without_invalid_launches() {
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    register_backend(Device::Cuda(0),b.clone());
    let dev=|s:&[usize]|Tensor::zeros(s).to_device(Device::Cuda(0)).unwrap();
    let x=dev(&[2,0,3]); let bias=dev(&[1,3]);
    let mut roots=vec![capture(||x.relu()),capture(||bias.sub(&x).unwrap()),capture(||x.transpose(0,2).unwrap()),capture(||x.sum_dim(0,false).unwrap())];
    for axis in 0..3 { roots.push(capture(||x.softmax(axis).unwrap())); }
    let a=dev(&[0,2]);let w=dev(&[2,3]);
    roots.push(capture(||a.matmul(&w).unwrap()));
    let a=dev(&[2,3]);let w=dev(&[3,0]);
    roots.push(capture(||a.matmul(&w).unwrap()));
    let a=dev(&[0,2,3]);let w=dev(&[0,3,4]);
    roots.push(capture(||a.bmm(&w).unwrap()));
    for root in roots {
        let mut graph=CompiledChain::compile(&root).unwrap().prepare_static().unwrap();
        for count in 1..=3 {
            let before=b.alloc_stats().requests;
            graph.replay().unwrap();assert_eq!(graph.replay_count(),count);
            assert_eq!(b.alloc_stats().requests,before);
            let snapshot=graph.snapshot().unwrap();assert_eq!(snapshot.device(),Device::Cuda(0));
            assert_eq!(snapshot.shape(),root.shape());assert!(snapshot.to_vec().is_empty());
        }
    }
}

#[test]
fn static_zero_reductions_overwrite_dirty_destinations_on_every_replay() {
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    use ferro_core::dispatch::{Backend,StaticRun,StaticOp,DeviceBuffer};
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    register_backend(Device::Cuda(0),b.clone());
    let empty:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[]).unwrap());
    for op in [StaticOp::MatMul {m:2,k:0,n:3},StaticOp::Bmm {batch:2,m:1,k:0,n:3}] {
        let dirty=b.alloc_from_host(&[99.;6]).unwrap();drop(dirty);
        let runs=[StaticRun {inputs:vec![0,0],shape:vec![2,3],op}];
        let mut graph=Backend::prepare_static(b.clone(),&runs,vec![empty.clone()]).unwrap();
        for _ in 0..3 {
            graph.replay().unwrap();
            let snapshot=graph.snapshot().unwrap();assert_eq!(b.copy_to_host(snapshot.as_ref()).unwrap(),vec![0.;6]);
            b.fill_inplace_dev(snapshot.as_ref(),77.).unwrap();
        }
    }
    let dev=|s:&[usize]|Tensor::zeros(s).to_device(Device::Cuda(0)).unwrap();
    let a=dev(&[2,0]);let w=dev(&[0,3]);let bias=Tensor::full(&[3],2.).to_device(Device::Cuda(0)).unwrap();
    let root=capture(||a.matmul(&w).unwrap().add(&bias).unwrap());
    let mut graph=CompiledChain::compile(&root).unwrap().prepare_static().unwrap();
    for v in [2.,3.] {
        bias.copy_from(&Tensor::full(&[3],v).to_device(Device::Cuda(0)).unwrap()).unwrap();
        graph.replay().unwrap();assert_eq!(graph.snapshot().unwrap().to_vec(),vec![v;6]);
    }
    let a=dev(&[2,1,0]);let w=dev(&[2,0,3]);
    let root=capture(||a.bmm(&w).unwrap());
    let mut graph=CompiledChain::compile(&root).unwrap().prepare_static().unwrap();
    graph.replay().unwrap();assert_eq!(graph.snapshot().unwrap().to_vec(),vec![0.;6]);
    let x=dev(&[2,0,3]);let root=capture(||x.sum_dim(1,false).unwrap());
    let mut graph=CompiledChain::compile(&root).unwrap().prepare_static().unwrap();
    graph.replay().unwrap();assert_eq!(graph.snapshot().unwrap().to_vec(),vec![0.;6]);
}

#[test]
fn empty_static_plans_still_validate_extents_strides_and_operands() {
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    use ferro_core::dispatch::{Backend,StaticRun,StaticOp,DeviceBuffer,ChainStepRef,BinaryKind};
    let b=match CudaBackend::new(0) { Ok(b)=>Arc::new(b), Err(e) if std::env::var_os("FERRO_REQUIRE_CUDA").is_some()=>panic!("required CUDA: {e}"), Err(_)=>return };
    let empty:Arc<dyn DeviceBuffer>=Arc::from(b.alloc_from_host(&[]).unwrap());
    let cases=[
        StaticRun {inputs:vec![0],shape:vec![0],op:StaticOp::Softmax {rows:usize::MAX,cols:0}},
        StaticRun {inputs:vec![0,0,0],shape:vec![0],op:StaticOp::LayerNorm {rows:0,cols:usize::MAX,eps:1e-5,weight:false,bias:false}},
        StaticRun {inputs:vec![0],shape:vec![0,u32::MAX as usize,u32::MAX as usize,2],op:StaticOp::Layout {strides:vec![0,0,2,1]}},
        StaticRun {inputs:vec![0],shape:vec![0,usize::MAX,2],op:StaticOp::Layout {strides:vec![0,2,1]}},
        StaticRun {inputs:vec![0],shape:vec![0,3],op:StaticOp::Broadcast {strides:vec![1]}},
        StaticRun {inputs:vec![0],shape:vec![0],op:StaticOp::Pointwise(vec![ChainStepRef::Binary {kind:BinaryKind::Sub,other:usize::MAX}])},
        StaticRun {inputs:vec![0,0],shape:vec![1],op:StaticOp::MatMul {m:1,k:0,n:2}},
        StaticRun {inputs:vec![0],shape:vec![usize::MAX,2],op:StaticOp::Layout {strides:vec![2,1]}},
    ];
    for run in cases {
        let before=b.alloc_stats().requests;
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||Backend::prepare_static(b.clone(),&[run],vec![empty.clone()])));
        assert!(result.is_ok(),"malformed plan must not panic");
        assert!(result.unwrap().is_err(),"malformed plan accepted");
        assert_eq!(b.alloc_stats().requests,before);
    }
}

// Backend registration is process-global; hold this through all tensor drops.
static REGISTRY: Mutex<()> = Mutex::new(());

#[test]
fn malformed_static_plan_is_rejected_without_panicking_or_allocating() {
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
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
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
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
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
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
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
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
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
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
    for value in [1.,-1.,2.] {
        w.copy_from(&Tensor::full(&[3,2], value).to_device(Device::Cuda(0)).unwrap()).unwrap();
        graph.replay().unwrap();
        assert_eq!(graph.snapshot().unwrap().to_vec(), compiled.replay().unwrap().to_vec());
    }
    let old = graph.snapshot().unwrap();
    drop(compiled); drop(root); drop(x); drop(w);
    graph.replay().unwrap();
    assert_eq!(graph.snapshot().unwrap().to_vec(), vec![12.,12.,30.,30.]);
    drop(graph);
    assert_eq!(old.to_vec(), vec![12.,12.,30.,30.]);
}

use ferro_core::{capture, Device, Tensor};
use ferro_core::dispatch::register_backend;
use ferro_core::graph::CompiledChain;
use ferro_cuda::CudaBackend;

#[test]
fn compiled_static_pointwise_reads_current_leaves_and_retains_snapshots() {
    let _registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
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
    drop(compiled); drop(root); drop(x); drop(w);
    graph.replay().unwrap();
    assert_eq!(graph.snapshot().unwrap().to_vec(), vec![4.,-8.,18.,-24.]);
    drop(graph);
    assert_eq!(old.to_vec(), vec![-2.,8.,-12.,24.]);
}
