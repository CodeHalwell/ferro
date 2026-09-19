use ferro_core::{Device, Tensor};
use ferro_core::segment::PreparedSegments;
use ferro_core::sparse::Coo;
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
fn init() -> bool {
    if let Err(e)=ferro_cuda::install(0) { assert!(std::env::var_os("FERRO_REQUIRE_CUDA").is_none(), "CUDA required: {e}"); false } else { true }
}
fn t(v: &[f32], s: &[usize], dev: Device) -> Tensor { Tensor::from_vec(v.to_vec(),s).unwrap().to_device(dev).unwrap().requires_grad_(true).unwrap() }
fn close(a:&Tensor,b:&Tensor) {
    assert_eq!(a.shape(),b.shape());
    for (a,b) in a.to_vec().iter().zip(b.to_vec()) { assert!((a-b).abs()<2e-5*(1.+b.abs()), "{a} != {b}"); }
}
#[test]
fn sparse_forward_backward_strided_inputs_and_seeds_stay_resident() {
    let _lock=LOCK.lock().unwrap_or_else(|e|e.into_inner()); if !init() {return;}
    let backend=ferro_cuda::cuda_backend().unwrap(); let d=Device::Cuda(0);
    let a=Coo::new(4,3,vec![2,0,2,1,0],vec![1,2,1,0,0]).unwrap();
    let before=backend.layout_counts();
    let p=a.prepare(d).unwrap(); let indices=backend.layout_counts();
    assert_eq!(indices.0-before.0,2);
    for shift in [0.,0.5] {
        let w=t(&[0.4,-0.3,0.2,0.6,0.1], &[5],d);
        let x=t(&[0.1+shift,0.2,0.3,0.4,0.5,0.6], &[2,3],d);
        let seed=t(&[0.2,0.4,0.6,0.8,0.1,0.3,0.5,0.7], &[2,4],d).transpose(0,1).unwrap();
        let transfers=backend.layer_norm_counts();
        let y=p.spmm(&w,&x.transpose(0,1).unwrap()).unwrap(); y.backward_with(&seed);
        let after=backend.layer_norm_counts();
        assert_eq!((after.1-transfers.1,after.2-transfers.2),(0,0)); assert_eq!((backend.layout_counts().0,backend.layout_counts().1),(indices.0,indices.1));
        let cw=t(&w.to_vec(), &[5],Device::Cpu); let cx=t(&x.to_vec(), &[2,3],Device::Cpu);
        let cy=a.spmm(&cw,&cx.transpose(0,1).unwrap()).unwrap(); cy.backward_with(&seed.to_device(Device::Cpu).unwrap());
        close(&y,&cy); close(&w.grad().unwrap(),&cw.grad().unwrap()); close(&x.grad().unwrap(),&cx.grad().unwrap());
        let l=t(&[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], &[2,4],d);
        let r=t(&[0.2,0.3,0.4,0.5,0.6,0.7], &[2,3],d);
        let g=t(&[0.3,0.2,-0.1,0.5,0.4], &[5],d);
        let transfers=backend.layer_norm_counts();
        let y=p.sddmm(&l.transpose(0,1).unwrap(),&r.transpose(0,1).unwrap()).unwrap(); y.backward_with(&g);
        let after=backend.layer_norm_counts();
        assert_eq!((after.1-transfers.1,after.2-transfers.2),(0,0)); assert_eq!((backend.layout_counts().0,backend.layout_counts().1),(indices.0,indices.1));
        let cl=t(&l.to_vec(), &[2,4],Device::Cpu); let cr=t(&r.to_vec(), &[2,3],Device::Cpu);
        let cy=a.sddmm(&cl.transpose(0,1).unwrap(),&cr.transpose(0,1).unwrap()).unwrap(); cy.backward_with(&g.to_device(Device::Cpu).unwrap());
        close(&y,&cy); close(&l.grad().unwrap(),&cl.grad().unwrap()); close(&r.grad().unwrap(),&cr.grad().unwrap());
        println!("prepared sparse repeat: topology uploads=0 feature uploads=0 feature downloads=0 in forward+backward");
    }
}
#[test]
fn arbitrary_axis_selection_scatter_and_reshape_adjoint_stay_resident() {
    let _lock=LOCK.lock().unwrap_or_else(|e|e.into_inner()); if !init() {return;}
    let b=ferro_cuda::cuda_backend().unwrap(); let d=Device::Cuda(0);
    let p=PreparedSegments::new(&[2,0,2],4,d).unwrap();
    for scatter in [false,true] {
        let x=t(&[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], &[2,4],d);
        let src=t(&[0.2,0.4,0.6,0.1,0.3,0.5], &[3,2],d);
        let shape=if scatter {[4,2]} else {[3,2]};
        let seed=Tensor::full_on(&shape,0.3,d).unwrap().transpose(0,1).unwrap();
        let before=b.layer_norm_counts(); let idx=b.layout_counts();
        let y=if scatter {p.scatter_add(&x,&src.transpose(0,1).unwrap(),1)} else {p.select(&x,1)}.unwrap();
        y.reshape(y.shape()).unwrap().backward_with(&seed);
        let after=b.layer_norm_counts(); assert_eq!((after.1-before.1,after.2-before.2),(0,0)); assert_eq!((b.layout_counts().0,b.layout_counts().1),(idx.0,idx.1));
        let cx=t(&x.to_vec(), &[2,4],Device::Cpu); let cs=t(&src.to_vec(), &[3,2],Device::Cpu);
        let cp=PreparedSegments::new(&[2,0,2],4,Device::Cpu).unwrap();
        let cy=if scatter {cp.scatter_add(&cx,&cs.transpose(0,1).unwrap(),1)} else {cp.select(&cx,1)}.unwrap();
        cy.backward_with(&seed.to_device(Device::Cpu).unwrap()); close(&y,&cy); close(&x.grad().unwrap(),&cx.grad().unwrap());
        if scatter { close(&src.grad().unwrap(),&cs.grad().unwrap()); }
    }
    // Convenience path uploads topology per call, but no feature transfers.
    let x=t(&[1.,2.,3.,4.,5.,6.,7.,8.], &[2,4],d);
    let seed=Tensor::full_on(&[2,3],1.,d).unwrap(); let before=b.layer_norm_counts();
    x.index_select(1,&[2,0,2]).unwrap().backward_with(&seed);
    let after=b.layer_norm_counts(); assert_eq!((after.1-before.1,after.2-before.2),(0,0));
}
