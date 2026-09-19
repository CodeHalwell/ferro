use ferro_core::{Device, Tensor};
static REGISTRY: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn prepared_topology_reuse_counts_and_empty_contracts() {
    use ferro_core::segment::PreparedSegments;
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = ferro_cuda::install(0) {
        assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}");
        return;
    }
    let b = ferro_cuda::cuda_backend().unwrap();
    let device = Device::Cuda(0);
    let idx = b.layout_counts();
    let plan = PreparedSegments::new(&[2,0,2], 4, device).unwrap();
    let prepared = b.layout_counts();
    assert_eq!((prepared.0-idx.0, prepared.1-idx.1), (1,64));
    for shift in [0., 2.] {
        let x = Tensor::from_vec(vec![1.+shift,2.,3.,4.,5.,6.], &[3,2]).unwrap().to_device(device).unwrap().requires_grad_(true).unwrap();
        let seed = Tensor::from_vec(vec![0.2; 6], &[3,2]).unwrap().to_device(device).unwrap();
        let sum_seed = Tensor::from_vec(vec![0.3; 8], &[4,2]).unwrap().to_device(device).unwrap();
        let c = b.segment_counts();
        let transfers = b.layer_norm_counts();
        let alloc = b.alloc_stats();
        let y = plan.sum(&x).unwrap();
        y.backward_with(&sum_seed);
        let z = plan.softmax(&x).unwrap();
        z.backward_with(&seed);
        let after = b.segment_counts();
        assert_eq!(std::array::from_fn::<_,6,_>(|i| after[i]-c[i]), [1,1,1,1,1,1]);
        let t = b.layer_norm_counts();
        assert_eq!((t.1-transfers.1,t.2-transfers.2), (0,0));
        assert_eq!(b.layout_counts(), prepared);
        // Four operator results; accumulated leaf gradient updates in place.
        assert_eq!(b.alloc_stats().requests - alloc.requests, 4);
        assert_eq!(x.grad().unwrap().device(), device);
        assert!((y.to_vec()[4] - (6.+shift)).abs() < 1e-6);
        println!("prepared step: index uploads=0 feature transfers=0 sum/adjoint=1/1 softmax/adjoint=1/1 control reads=1 (4 bytes), f32 allocations=4, control allocations=1");
    }
    for (shape, ids, groups) in [(vec![0,2], vec![], 4), (vec![0,2], vec![], 0), (vec![3,0], vec![2,0,2], 4)] {
        let p = PreparedSegments::new(&ids, groups, device).unwrap();
        let x = Tensor::from_vec(vec![], &shape).unwrap().to_device(device).unwrap().requires_grad_(true).unwrap();
        let y = p.sum(&x).unwrap();
        assert!(y.to_vec().iter().all(|&v| v == 0.));
        y.backward_with(&Tensor::full_on(y.shape(), 1., device).unwrap());
        assert!(x.grad().unwrap().to_vec().is_empty());
        assert_eq!(p.softmax(&x).unwrap().shape(), shape);
    }
    let x = Tensor::from_vec(vec![f32::NEG_INFINITY], &[1,1]).unwrap().to_device(device).unwrap();
    assert!(ferro_core::segment::softmax(&x, &[0], 1).is_err());
    for v in [f32::INFINITY, f32::NAN] {
        let x = Tensor::from_vec(vec![v], &[1]).unwrap().to_device(device).unwrap();
        assert!(ferro_core::segment::softmax(&x, &[0], 1).is_err());
    }
    assert!(PreparedSegments::new(&[4], 4, device).is_err());
    assert!(PreparedSegments::new(&[], usize::MAX, device).is_err());
    assert!(ferro_core::segment::mean(&x, &[0], 1).is_err());
    assert!(ferro_core::segment::max(&x, &[0], 1).is_err());
    let _ = ferro_core::capture(|| {
        assert!(plan.sum(&x).is_err());
        assert!(plan.softmax(&x).is_err());
        x.clone()
    });
}

// Independent f64 scalar oracle, original edge order and explicit VJP.
fn oracle(data: &[f64], ids: &[usize], groups: usize, width: usize, soft: bool) -> Vec<f64> {
    if !soft {
        let mut y = vec![0.; groups*width];
        for (e,&s) in ids.iter().enumerate() { for f in 0..width { y[s*width+f] += data[e*width+f]; } }
        return y;
    }
    let mut y = vec![0.; data.len()];
    for s in 0..groups { for f in 0..width {
        let edges: Vec<_> = ids.iter().enumerate().filter_map(|(e,&id)| (id==s).then_some(e)).collect();
        let m = edges.iter().map(|&e| data[e*width+f]).fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = edges.iter().map(|&e| (data[e*width+f]-m).exp()).sum();
        for e in edges { y[e*width+f] = (data[e*width+f]-m).exp()/z; }
    } }
    y
}

#[test]
fn f64_parity_and_finite_difference_vjp() {
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = ferro_cuda::install(0) {
        assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}"); return;
    }
    for (edges, width, groups) in [(7,3,5), (259,17,31), (513,5,1)] {
        let ids: Vec<_> = (0..edges).map(|i| (i*17+i/3)%groups).collect();
        let data: Vec<f32> = (0..edges*width).map(|i| ((i*31%127) as f32-63.)/31.).collect();
        for soft in [false,true] {
            let x = Tensor::from_vec(data.clone(), &[edges,width]).unwrap().to_device(Device::Cuda(0)).unwrap().requires_grad_(true).unwrap();
            let p = ferro_core::segment::PreparedSegments::new(&ids,groups,x.device()).unwrap();
            let y = if soft { p.softmax(&x) } else { p.sum(&x) }.unwrap();
            let values: Vec<f64> = data.iter().map(|&x| x as f64).collect();
            let expected = oracle(&values, &ids, groups, width, soft);
            for (a,b) in y.to_vec().iter().zip(&expected) { assert!((*a as f64-b).abs() < 3e-5*(1.+b.abs()), "{a} != {b}"); }
            let seed: Vec<f32> = (0..expected.len()).map(|i| (i%11) as f32/7.-0.5).collect();
            y.backward_with(&Tensor::from_vec(seed.clone(),y.shape()).unwrap().to_device(x.device()).unwrap());
            let actual = x.grad().unwrap().to_vec();
            let mut dots = vec![0.;groups*width];
            if soft { for (e,&s) in ids.iter().enumerate() { for f in 0..width { dots[s*width+f] += expected[e*width+f]*seed[e*width+f] as f64; } } }
            for (e,&s) in ids.iter().enumerate() { for f in 0..width {
                let j=e*width+f;
                let dx=if soft { expected[j]*(seed[j] as f64-dots[s*width+f]) } else { seed[s*width+f] as f64 };
                assert!((actual[j] as f64-dx).abs() < 3e-6, "gradient {j}: {} != {dx}",actual[j]);
            } }
            if edges == 7 {
                for j in 0..values.len() {
                    let mut plus=values.clone(); let mut minus=values.clone();
                    plus[j]+=1e-4; minus[j]-=1e-4;
                    let a=oracle(&plus,&ids,groups,width,soft); let b=oracle(&minus,&ids,groups,width,soft);
                    let fd: f64=a.iter().zip(b).zip(&seed).map(|((&a,b),&s)| (a-b)*s as f64/2e-4).sum();
                    assert!((actual[j] as f64-fd).abs()<3e-6);
                    // Also exercise perturbed public GPU forwards, not just the oracle.
                    let xp=Tensor::from_vec(plus.iter().map(|&v|v as f32).collect(),x.shape()).unwrap().to_device(x.device()).unwrap();
                    let xm=Tensor::from_vec(minus.iter().map(|&v|v as f32).collect(),x.shape()).unwrap().to_device(x.device()).unwrap();
                    let yp=if soft {p.softmax(&xp)} else {p.sum(&xp)}.unwrap().to_vec();
                    let ym=if soft {p.softmax(&xm)} else {p.sum(&xm)}.unwrap().to_vec();
                    let fd_gpu: f64=yp.iter().zip(ym).zip(&seed).map(|((&a,b),&s)| (a as f64-b as f64)*s as f64/2e-4).sum();
                    assert!((actual[j] as f64-fd_gpu).abs()<2e-3);
                }
            }
        }
    }
}

#[test]
fn strided_features_remain_resident() {
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = ferro_cuda::install(0) {
        assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}"); return;
    }
    let b = ferro_cuda::cuda_backend().unwrap();
    let base = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], &[2,3]).unwrap();
    let x = base.to_device(Device::Cuda(0)).unwrap().requires_grad_(true).unwrap();
    let view = x.transpose(0,1).unwrap();
    let ids = [1,0,1];
    let p = ferro_core::segment::PreparedSegments::new(&ids,2,x.device()).unwrap();
    let seed = view.clone();
    let before = b.layer_norm_counts();
    let y = p.softmax(&view).unwrap();
    y.backward_with(&seed);
    let after = b.layer_norm_counts();
    assert_eq!((after.1-before.1,after.2-before.2),(0,0));
    let cpu = base.requires_grad_(true).unwrap();
    let cv = cpu.transpose(0,1).unwrap();
    let cy = ferro_core::segment::softmax(&cv,&ids,2).unwrap();
    cy.backward_with(&cv.detach_copy());
    for (a,b) in y.to_vec().iter().zip(cy.to_vec()) { assert!((a-b).abs()<2e-6); }
    for (a,b) in x.grad().unwrap().to_vec().iter().zip(cpu.grad().unwrap().to_vec()) { assert!((a-b).abs()<2e-6); }
}

#[test]
fn backend_validation_stream_identity_and_plan_lifetime() {
    use ferro_core::dispatch::{Backend, SegmentOp};
    let b = match ferro_cuda::CudaBackend::new(0) {
        Ok(b) => b,
        Err(e) => { assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}"); return; }
    };
    let other = ferro_cuda::CudaBackend::new(0).unwrap();
    let plan = b.prepare_segments(&[1,0,1],3).unwrap();
    let x = b.alloc_from_host(&[1.,2.,3.]).unwrap();
    let wrong = b.alloc_from_host(&[1.,2.]).unwrap();
    let foreign = other.alloc_from_host(&[1.,2.,3.]).unwrap();
    let foreign_plan = other.prepare_segments(&[1,0,1],3).unwrap();
    let idx = b.alloc_i64_from_host(&[1,2,3]).unwrap();
    let c=b.segment_counts();
    assert!(b.segment_dev(plan.as_ref(),SegmentOp::Sum,wrong.as_ref(),None,1).is_err());
    assert!(b.segment_dev(plan.as_ref(),SegmentOp::Sum,foreign.as_ref(),None,1).is_err());
    assert!(b.segment_dev(foreign_plan.as_ref(),SegmentOp::Sum,x.as_ref(),None,1).is_err());
    assert!(b.segment_dev(plan.as_ref(),SegmentOp::Sum,idx.as_ref(),None,1).is_err());
    assert!(b.segment_dev(plan.as_ref(),SegmentOp::Sum,x.as_ref(),None,usize::MAX).is_err());
    assert!(b.segment_dev(plan.as_ref(),SegmentOp::SoftmaxBackward,x.as_ref(),None,1).is_err());
    assert!(b.segment_dev(plan.as_ref(),SegmentOp::SoftmaxBackward,x.as_ref(),Some(wrong.as_ref()),1).is_err());
    assert!(b.segment_dev(plan.as_ref(),SegmentOp::SoftmaxBackward,x.as_ref(),Some(foreign.as_ref()),1).is_err());
    assert_eq!(b.segment_counts(),c);
    let out=b.segment_dev(plan.as_ref(),SegmentOp::Sum,x.as_ref(),None,1).unwrap();
    drop(plan); drop(x);
    assert_eq!(b.copy_to_host(out.as_ref()).unwrap(),vec![2.,4.,0.]);
}

#[test]
fn public_softmax_stable_and_adjoint() {
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = ferro_cuda::install(0) {
        assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}");
        return;
    }
    let data = vec![1000.,-1000.,1001.,-999.,1002.,-1002.];
    let ids = [2,0,2];
    let x = Tensor::from_vec(data.clone(), &[3,2]).unwrap().to_device(Device::Cuda(0)).unwrap().requires_grad_(true).unwrap();
    let cpu = Tensor::from_vec(data, &[3,2]).unwrap().requires_grad_(true).unwrap();
    let expected = ferro_core::segment::softmax(&cpu, &ids, 4).unwrap();
    let y = ferro_core::segment::softmax(&x, &ids, 4).unwrap();
    assert_eq!(y.device(), x.device());
    for (a,b) in y.to_vec().iter().zip(expected.to_vec()) { assert!((a-b).abs() < 2e-6); }
    let seed = Tensor::from_vec(vec![0.2,0.4,0.5,-0.2,0.7,0.3], &[3,2]).unwrap();
    expected.backward_with(&seed);
    y.backward_with(&seed.to_device(x.device()).unwrap());
    let g = x.grad().unwrap();
    assert_eq!(g.device(), x.device());
    for (a,b) in g.to_vec().iter().zip(cpu.grad().unwrap().to_vec()) { assert!((a-b).abs() < 2e-6); }
}

#[test]
fn public_sum_is_resident_with_resident_adjoint() {
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = ferro_cuda::install(0) {
        assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}");
        return;
    }
    let x = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.], &[3,2]).unwrap()
        .to_device(Device::Cuda(0)).unwrap().requires_grad_(true).unwrap();
    let y = ferro_core::segment::sum(&x, &[2,0,2], 4).unwrap();
    assert_eq!(y.device(), x.device());
    assert_eq!(y.to_vec(), vec![3.,4.,0.,0.,6.,8.,0.,0.]);
    let seed = Tensor::from_vec(vec![1.,2.,3.,4.,5.,6.,7.,8.], &[4,2]).unwrap().to_device(x.device()).unwrap();
    y.backward_with(&seed);
    let g = x.grad().unwrap();
    assert_eq!(g.device(), x.device());
    assert_eq!(g.to_vec(), vec![5.,6.,1.,2.,5.,6.]);
}
