use ferro_core::{capture, Device, Tensor};
use ferro_core::graph::CompiledChain;

fn host(start: usize) -> Tensor {
    Tensor::from_vec((start..start+24).map(|v| v as f32 / 8.).collect(), &[2,3,4]).unwrap()
}
fn build(x: &Tensor, case: usize) -> Tensor {
    match case {
        0 => x.transpose(0,1).unwrap().neg(),
        1 => x.reshape(&[6,4]).unwrap().neg(),
        2 => {
            let t=capture(||x.transpose(0,2).unwrap().transpose(1,2).unwrap());
            let r=t.reshape(&[4,6]).unwrap();
            r.neg().add(&r).unwrap().add(&r.neg()).unwrap()
        }
        _ => unreachable!(),
    }
}
#[test]
fn cuda_capture_layout_updates_and_alias_safety() {
    ferro_cuda::install(0).expect("real CUDA required; this verification must not skip");
    let dev=Device::Cuda(0);
    for retained in [true,false] {
        for case in 0..3 {
            let x=host(1).to_device(dev).unwrap();
            let root=capture(||build(&x,case));
            let original=root.to_vec();
            let graph=CompiledChain::compile(&root).unwrap();
            let root=if retained {Some(root)} else {drop(root);None};
            for start in [25,49] {
                x.copy_from(&host(start).to_device(dev).unwrap()).unwrap();
                let replay=capture(||graph.replay().unwrap());
                assert_eq!(replay.device(),dev);
                assert!(!replay.requires_grad());
                assert_eq!(replay.to_vec(),build(&host(start),case).to_vec());
                if let Some(root)=&root {assert_eq!(root.to_vec(),original);}
            }
            println!("CUDA case={case} retained={retained}: repeated input update + replay passed");
        }
    }
    // Device ordinary views and detached snapshots must STILL reject writes,
    // even when an unrelated inference tape is retained or has been compiled.
    for case in 0..3 {
        let x=host(1).to_device(dev).unwrap();
        let alias=match case {
            0=>x.transpose(0,1).unwrap(),
            1=>x.reshape(&[6,4]).unwrap(),
            _=>x.detach_copy(),
        };
        let before=alias.to_vec();
        let root=capture(||build(&x,0));
        let _compiled=CompiledChain::compile(&root).unwrap();
        assert!(x.copy_from(&host(25).to_device(dev).unwrap()).is_err());
        assert_eq!(alias.to_vec(),before);
        drop(alias);
        x.copy_from(&host(25).to_device(dev).unwrap()).unwrap();
    }
    // A captured layout on a grad input keeps the ordinary view/version tape.
    let x=host(1).to_device(dev).unwrap().requires_grad_(true).unwrap();
    let root=capture(||x.transpose(0,1).unwrap().reshape(&[24]).unwrap().neg());
    assert!(root.requires_grad());
    assert!(x.copy_from(&host(25).to_device(dev).unwrap()).is_err());
    root.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(),vec![-1.;24]);
    // A saved OUTPUT snapshot must also remain immune to public writes.
    let x=host(1).to_device(dev).unwrap().requires_grad_(true).unwrap();
    let y=capture(||x.exp());
    let detached=y.detach_copy();
    let before=detached.to_vec();
    assert!(detached.copy_from(&host(25).to_device(dev).unwrap()).is_err());
    y.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(),before);
    println!("CUDA ordinary transpose/reshape/detach aliases, captured backward and output snapshot protections passed");
}
