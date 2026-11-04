use ferro_core::{Tensor,DType};
fn leaf(x:f32)->Tensor {Tensor::scalar(x).requires_grad_(true).unwrap()}

#[test]
fn finite_shared_roots_cancel_before_any_overflowing_adjoint() {
    for dtype in [DType::F32,DType::F64] {
        let (x,p,n)=if dtype==DType::F32 {(leaf(1e10),leaf(1e30),leaf(-1e30))} else {
            let make=|value|Tensor::from_vec_f64(vec![value],&[]).unwrap().requires_grad_(true).unwrap();
            (make(1e100),make(1e250),make(-1e250))
        };
        let shared=x.mul(&x).unwrap();
        let values=if dtype==DType::F32 {shared.to_vec().into_iter().map(f64::from).collect::<Vec<_>>()}else{shared.to_vec_f64()};
        assert!(values.iter().all(|value|value.is_finite()));
        for graph in [false,true] {
            if graph && dtype==DType::F64 {continue;}
            let got=Tensor::vjp_many_wrt_transform(&[&shared,&shared],&[&x],&[&p,&n],graph,false,None).unwrap().remove(0).unwrap();
            let values=if dtype==DType::F32 {got.to_vec().into_iter().map(f64::from).collect::<Vec<_>>()}else{got.to_vec_f64()};
            assert_eq!(values,vec![0.]);
            let separate=shared.vjp_wrt(&[&x],&p,false).unwrap()[0].add(&shared.vjp_wrt(&[&x],&n,false).unwrap()[0]).unwrap();
            let values=if dtype==DType::F32 {separate.to_vec().into_iter().map(f64::from).collect::<Vec<_>>()}else{separate.to_vec_f64()};
            assert!(values[0].is_nan());
        }
        assert!(x.grad().is_none()&&p.grad().is_none()&&n.grad().is_none());
    }
}

#[test]
fn disconnected_output_cannot_destroy_passthrough_identity_or_materialize_absence() {
    let x=leaf(2.);let unrelated=leaf(3.);let unused=leaf(4.);let seed=leaf(7.);let other=leaf(9.);
    for order in [false,true] {
        let outputs=if order{vec![&unrelated,&x]}else{vec![&x,&unrelated]};
        let seeds=if order{vec![&other,&seed]}else{vec![&seed,&other]};
        let result=Tensor::vjp_many_wrt_transform(&outputs,&[&x,&unused],&seeds,false,true,None).unwrap();
        assert_eq!(result[0].as_ref().unwrap().id(),seed.id());assert!(result[1].is_none());
        let detached=Tensor::vjp_many_wrt_transform(&outputs,&[&x],&seeds,false,false,None).unwrap();
        assert_eq!(detached[0].as_ref().unwrap().item(),7.);assert!(!detached[0].as_ref().unwrap().requires_grad());
        assert_ne!(detached[0].as_ref().unwrap().id(),seed.id());
    }
}

#[test]
fn frozen_joint_pullback_retains_only_original_primals_and_live_seed_dependencies() {
    let x=leaf(2.);let h=x.mul(&x).unwrap();let cutoff=h.id();let out=h.mul(&h).unwrap();
    let p=leaf(3.);let n=leaf(5.);
    let got=Tensor::vjp_many_wrt_transform(&[&out,&out],&[&x],&[&p,&n],true,false,Some((cutoff,&[]))).unwrap().remove(0).unwrap();
    assert_eq!(got.item(),256.);
    let next=got.grad_wrt(&[&x,&p,&n],false).unwrap();
    assert_eq!(next.iter().map(Tensor::item).collect::<Vec<_>>(),vec![128.,32.,32.]);
    // Actual Torch no-grad-created func.vjp gives these mixed partials.
    assert!(Tensor::vjp_many_wrt_transform(&[&out],&[&x],&[&p],false,false,Some((cutoff,&[]))).is_err());
    assert!(Tensor::vjp_many_wrt_transform(&[&out],&[&x],&[&p],true,false,Some((cutoff,&[&n]))).is_err());
    assert!(x.grad().is_none()&&p.grad().is_none()&&n.grad().is_none());
}

#[test]
fn active_reverse_frame_uses_joint_roots_without_changing_public_query_policy() {
    use ferro_core::forward_ad::{DualLevel,ReverseFrameGuard};
    let x=leaf(1e10);let direction=Tensor::scalar(3.);let positive=leaf(1e30);let negative=leaf(-1e30);
    assert!(Tensor::vjp_many_wrt_forward(&[&x],&[&x],&[&positive]).is_err());
    let level=DualLevel::new(&[&x],&[&direction]).unwrap();
    let _frame=ReverseFrameGuard::new().unwrap();
    let shared=x.mul(&x).unwrap();
    assert!(Tensor::vjp_many_wrt_transform(&[&shared],&[&x],&[&positive],true,false,None).is_err());
    let got=Tensor::vjp_many_wrt_forward(&[&shared,&shared],&[&x],&[&positive,&negative]).unwrap().remove(0).unwrap();
    assert_eq!(got.item(),0.);assert_eq!(level.tangent(&got).unwrap().unwrap().item(),0.);
    assert!(x.grad().is_none());
}

#[test]
fn distinct_roots_visit_the_shared_native_callback_once() {
    use std::sync::{Arc,atomic::{AtomicUsize,Ordering}};
    let count=Arc::new(AtomicUsize::new(0));let seen=count.clone();let x=leaf(2.);
    let shared=Tensor::scalar(2.*x.item()-1.).record_fn(vec![x.clone()],move|g|{
        seen.fetch_add(1,Ordering::Relaxed);vec![g.mul(&Tensor::scalar(2.)).unwrap()]
    });
    let left=shared.mul(&Tensor::scalar(3.)).unwrap();let right=shared.add(&Tensor::scalar(5.)).unwrap();
    let positive=Tensor::scalar(1.);let negative=Tensor::scalar(-3.);
    let result=Tensor::vjp_many_wrt_transform(&[&left,&right],&[&x],&[&positive,&negative],false,false,None).unwrap();
    assert_eq!(result[0].as_ref().unwrap().item(),0.);assert_eq!(count.load(Ordering::Relaxed),1);
    count.store(0,Ordering::Relaxed);
    let result=Tensor::vjp_many_wrt_transform(&[&left,&right],&[&shared],&[&positive,&negative],false,false,None).unwrap();
    assert_eq!(result[0].as_ref().unwrap().item(),0.);assert_eq!(count.load(Ordering::Relaxed),0);
}
