use ferro_core::{Backend, Device, Tensor};
use ferro_core::segment::PreparedSegments;
use ferro_cuda::CudaBackend;
use std::sync::Mutex;

static REGISTRY: Mutex<()> = Mutex::new(());
const DEV: Device = Device::Cuda(0);

fn counts(b: &CudaBackend) -> [usize; 4] {
    let t = b.layer_norm_counts();
    let l = b.layout_counts();
    [t.1, t.2, l.0, l.2]
}
fn delta(b: &CudaBackend, before: [usize; 4]) -> [usize; 4] {
    let after = counts(b);
    std::array::from_fn(|i| after[i] - before[i])
}
fn tensor(values: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(values, shape).unwrap().to_device(DEV).unwrap()
}

#[test]
fn required_cuda_strided_seed_detach_backward_and_identity_vjp() {
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    ferro_cuda::install(0).expect("real CUDA required; this test must not silently skip");
    let b = ferro_cuda::cuda_backend().unwrap();
    // Validate that the actual transfer counters observe a known round trip.
    let before = counts(&b);
    assert_eq!(tensor(vec![7.], &[1]).to_vec(), vec![7.]);
    assert_eq!(delta(&b, before), [1, 1, 0, 0]);
    for whole in [false, true] {
        for mode in ["detach", "backward", "vjp"] {
            let base = tensor(vec![1.,2.,3.,4.,5.,6.], &[3,2]).requires_grad_(true).unwrap();
            let seed = if whole { base.clone() } else { base.transpose(0,1).unwrap() };
            let root = tensor(vec![0.;6], seed.shape()).requires_grad_(true).unwrap();
            let before = counts(&b);
            let result = match mode {
                "detach" => seed.detach_copy(),
                "backward" => { root.backward_with(&seed); root.grad().unwrap() },
                _ => root.vjp_wrt(&[&root], &seed, false).unwrap().remove(0),
            };
            let measured = delta(&b, before);
            assert_eq!(measured, [0,0,0,usize::from(!whole)], "{mode} whole={whole}");
            assert_eq!(result.device(), DEV);
            assert!(!result.requires_grad());
            assert!(base.grad().is_none());
            drop(seed); drop(base);
            assert_eq!(result.to_vec(), if whole { vec![1.,2.,3.,4.,5.,6.] } else { vec![1.,3.,5.,2.,4.,6.] });
            println!("{mode} whole={whole}: [f32 upload, download, i64 upload, direct layout]={measured:?}");
        }
    }
}

#[test]
fn required_cuda_rank_boundary_fallback() {
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    ferro_cuda::install(0).expect("real CUDA required");
    let b = ferro_cuda::cuda_backend().unwrap();
    for rank in [17, 16] {
        for empty in [false, true] {
            for whole in [false, true] {
                for mode in ["detach", "backward", "vjp"] {
                    let mut shape = vec![1; rank];
                    shape[0] = 2; shape[1] = 3;
                    if empty { shape[0] = 0; }
                    let values = if empty { vec![] } else { vec![1.,2.,3.,4.,5.,6.] };
                    let base = tensor(values.clone(), &shape);
                    let seed = if whole { base.clone() } else { base.transpose(0,1).unwrap() };
                    let root = tensor(vec![0.; values.len()], seed.shape()).requires_grad_(true).unwrap();
                    let before = counts(&b);
                    let result = match mode {
                        "detach" => seed.detach_copy(),
                        "backward" => { root.backward_with(&seed); root.grad().unwrap() },
                        _ => root.vjp_wrt(&[&root], &seed, false).unwrap().remove(0),
                    };
                    let fallback = rank > 16 && !whole;
                    let measured = delta(&b, before);
                    assert_eq!(measured, [usize::from(fallback && mode == "backward"), usize::from(fallback), 0,
                        usize::from(!fallback && !whole && !empty)], "rank={rank} empty={empty} whole={whole} {mode}");
                    assert_eq!(result.device(), if fallback && mode != "backward" { Device::Cpu } else { DEV });
                    assert_eq!(result.shape(), seed.shape());
                    assert_eq!(result.to_vec(), if whole || empty { values } else { vec![1.,4.,2.,5.,3.,6.] });
                    if whole { assert_eq!(result._storage_ptr(), seed._storage_ptr()); }
                    println!("rank={rank} empty={empty} whole={whole} {mode}: {measured:?}");
                }
            }
        }
    }
}

#[test]
fn required_cuda_high_rank_invalid_layouts_do_not_decline() {
    use ferro_core::dispatch::MaterializeSupport;
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    ferro_cuda::install(0).expect("real CUDA required");
    let b = ferro_cuda::cuda_backend().unwrap();
    let buf = b.alloc_from_host(&[1.,2.,3.,4.,5.,6.]).unwrap();
    let mut shape = vec![1; 17];
    shape[0] = 3; shape[1] = 2;
    let mut strides = vec![1; 17];
    strides[1] = 3;
    let before = counts(&b);
    assert_eq!(b.materialize_support_for(&*buf, &shape, &strides, 0).unwrap(), MaterializeSupport::UnsupportedLayout);
    assert!(b.materialize_support_for(&*buf, &shape, &strides[..16], 0).is_err());
    assert!(b.materialize_support_for(&*buf, &shape, &strides, 1).is_err());
    assert!(b.materialize_support_for(&*buf, &shape, &strides, usize::MAX).is_err());
    shape[0] = usize::MAX;
    assert!(b.materialize_support_for(&*buf, &shape, &strides, 0).is_err());
    shape[0] = 0;
    assert!(b.materialize_support_for(&*buf, &shape, &strides, 7).is_err());
    assert_eq!(b.materialize_support_for(&*buf, &shape, &strides, 6).unwrap(), MaterializeSupport::UnsupportedLayout);
    assert_eq!(delta(&b, before), [0,0,0,0]);
}

#[test]
fn required_cuda_prepared_segments_strided_seeds() {
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    ferro_cuda::install(0).expect("real CUDA required");
    let b = ferro_cuda::cuda_backend().unwrap();
    let ids = [2,0,2];
    let p = PreparedSegments::new(&ids,4,DEV).unwrap();
    for soft in [false,true] {
        let data = vec![1.,2.,3.,4.,5.,6.];
        let x = tensor(data.clone(), &[3,2]).requires_grad_(true).unwrap();
        let cpu = Tensor::from_vec(data, &[3,2]).unwrap().requires_grad_(true).unwrap();
        let rows = if soft { 3 } else { 4 };
        let values: Vec<_> = (0..rows*2).map(|i| i as f32 / 7. - 0.3).collect();
        let host_seed = Tensor::from_vec(values.clone(), &[2,rows]).unwrap().transpose(0,1).unwrap();
        let seed = tensor(values, &[2,rows]).transpose(0,1).unwrap();
        let before = counts(&b);
        let sc = b.segment_counts();
        let y = if soft { p.softmax(&x) } else { p.sum(&x) }.unwrap();
        y.backward_with(&seed);
        let measured = delta(&b,before);
        assert_eq!(measured,[0,0,0,1]);
        let after = b.segment_counts();
        let sd: [usize;6] = std::array::from_fn(|i| after[i]-sc[i]);
        assert_eq!(sd, if soft { [0,0,1,1,1,1] } else { [1,1,0,0,0,0] });
        let cy = if soft { ferro_core::segment::softmax(&cpu,&ids,4) } else { ferro_core::segment::sum(&cpu,&ids,4) }.unwrap();
        cy.backward_with(&host_seed);
        for (a,c) in y.to_vec().iter().zip(cy.to_vec()) { assert!((a-c).abs()<2e-6); }
        for (a,c) in x.grad().unwrap().to_vec().iter().zip(cpu.grad().unwrap().to_vec()) { assert!((a-c).abs()<2e-6); }
        println!("prepared soft={soft}: transfers/layout={measured:?}, segment counters={sd:?}; softmax control read is 4 bytes, not a feature download");
    }
}

#[test]
fn required_cuda_bounded_registry_replacement() {
    let _lock = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    ferro_cuda::install(0).expect("real CUDA required");
    let old = ferro_cuda::cuda_backend().unwrap();
    let seed = tensor(vec![1.,2.,3.,4.,5.,6.], &[2,3]).transpose(0,1).unwrap();
    let mut high_shape = vec![1; 17];
    high_shape[0] = 2; high_shape[1] = 3;
    let high_seed = tensor(vec![1.,2.,3.,4.,5.,6.], &high_shape).transpose(0,1).unwrap();
    let root = tensor(vec![0.;6], &[3,2]).requires_grad_(true).unwrap();
    root.backward_with(&seed);
    let grad = root.grad().unwrap();
    let p = PreparedSegments::new(&[1,0,1],2,DEV).unwrap();
    let x = tensor(vec![1.,2.,3.,4.,5.,6.], &[3,2]);
    let y = p.sum(&x).unwrap();
    ferro_cuda::install(0).expect("replacement CUDA backend required");
    let new = ferro_cuda::cuda_backend().unwrap();
    drop(old);
    let before = counts(&new);
    assert_eq!(seed.to_vec(),vec![1.,4.,2.,5.,3.,6.]);
    assert_eq!(grad.to_vec(),vec![1.,4.,2.,5.,3.,6.]);
    assert_eq!(y.to_vec(),vec![3.,4.,6.,8.]);
    assert_eq!(delta(&new,before),[0,3,0,0]);
    let fresh = tensor(vec![1.;6], &[3,2]);
    assert!(p.sum(&fresh).err().expect("foreign compute must fail").to_string().contains("different CUDA backend stream/context"));
    for seed in [&seed, &high_seed] {
        let before = counts(&new);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| seed.detach_copy())).is_err());
        assert_eq!(delta(&new,before),[0,0,0,0], "foreign materialization must not fall back to host at rank {}", seed.shape().len());
    }
    let buffer = new.alloc_from_host(&[2.]).unwrap();
    assert_eq!(new.copy_to_host(&*buffer).unwrap(),vec![2.]);
    println!("replacement: old host reads supported; foreign prepared compute and strided detach rejected without host fallback");
}
