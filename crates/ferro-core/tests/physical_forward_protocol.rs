use ferro_core::{Tensor,DType};
use ferro_core::autograd::{with_grad_enabled,is_grad_enabled};
use ferro_core::forward_ad::{DualLevel,ReverseFrameGuard,current_tangents,prepare_alias,unwrap_values,post_alias_columns,is_active};
use std::sync::{Arc,atomic::{AtomicUsize,Ordering}};

fn tensor(data:&[f32],shape:&[usize])->Tensor {Tensor::from_vec(data.to_vec(),shape).unwrap()}
fn close(a:&[f32],b:&[f32]) {assert_eq!(a.len(),b.len());for(a,b)in a.iter().zip(b){assert!((a-b).abs()<3e-6,"{a}!={b}");}}

#[test]
fn physical_tangent_metadata_alias_public_boundaries_and_error_cleanup_are_atomic() {
    let x=tensor(&[0.2,-0.4],&[2]);
    let basis=tensor(&[1.,0.,0.,1.],&[2,2]);
    assert!(DualLevel::functional_batched(&[&x],&[&basis],3).is_err());
    assert!(DualLevel::functional_batched(&[&x],&[],2).is_err());
    assert!(!is_active());
    let level=DualLevel::functional_batched(&[&x],&[&basis],2).unwrap();
    assert!(DualLevel::functional(&[&x],&[&Tensor::ones(&[2])]).is_err());
    assert!(level.tangent(&x).is_err());
    assert!(x.copy_from(&Tensor::zeros(&[2])).is_err());
    let alias=x.detach();
    assert!(post_alias_columns(&x,&alias,Some(&[&Tensor::ones(&[2])])).is_err());
    assert!(post_alias_columns(&x,&alias,Some(&[&basis,&basis])).is_err());
    assert!(level.tangents(&alias).unwrap().is_none());
    post_alias_columns(&x,&alias,Some(&[&basis])).unwrap();
    assert_eq!(level.tangents(&alias).unwrap().unwrap()[0].id(),basis.id());
    let calls=Arc::new(AtomicUsize::new(0));let seen=calls.clone();
    let output=Tensor::ones(&[2]).record_fn(vec![x.clone()],move|_|{seen.fetch_add(1,Ordering::Relaxed);vec![Tensor::ones(&[3])]});
    assert!(level.tangents(&output).is_err());
    assert!(level.tangents(&x).is_err());
    assert_eq!(calls.load(Ordering::Relaxed),1);
    drop(level);
    assert!(is_grad_enabled()&&!is_active());
    let empty=Tensor::zeros(&[0,1usize<<60]);
    let wide=Tensor::from_vec_f64(vec![],&[2,0,1usize<<60]).unwrap();
    let level=DualLevel::functional_batched(&[&empty],&[&wide],2).unwrap();
    let output=empty.unsqueeze(0).unwrap();
    let tangent=level.tangents(&output).unwrap().unwrap().remove(0);
    assert_eq!(tangent.shape(),&[2,1,0,1usize<<60]);assert_eq!(tangent.dtype(),DType::F64);
    let sum=empty.sum();let sum_tangent=level.batched_tangent(&sum).unwrap().unwrap();
    assert_eq!(sum_tangent.shape(),&[2]);assert_eq!(sum_tangent.to_vec_f64(),vec![0.,0.]);
    drop(level);
    assert!(x.grad().is_none()&&basis.grad().is_none());
}

#[test]
fn whole_batch_reverse_frame_projection_preserves_captured_flags_and_no_grad_coefficients() {
    for live in [false,true] {
        let x=tensor(&[0.2,-0.4],&[2]).requires_grad_(live).unwrap();
        let w=Tensor::scalar(0.5).requires_grad_(true).unwrap();
        let basis=tensor(&[1.,0.,0.,1.],&[2,2]);
        let level=DualLevel::functional_batched(&[&x],&[&basis],2).unwrap();
        let frame=ReverseFrameGuard::new().unwrap();
        let root=if live{x.reshape(&[2]).unwrap()}else{x.detach().requires_grad_(true).unwrap()};
        prepare_alias(&x,&root).unwrap();
        let coefficient=with_grad_enabled(false,||root.mul(&root).unwrap().tanh());
        let loss=root.mul(&coefficient).unwrap().mul(&w).unwrap().sum();
        let gradient=loss.vjp_wrt_forward(&[&root],&Tensor::scalar(1.)).unwrap().remove(0).unwrap();
        let physical=current_tangents(&gradient).unwrap().unwrap().remove(0);
        let outputs=unwrap_values(&[&gradient,&physical,&physical]).unwrap();
        assert_eq!(outputs[1].id(),outputs[2].id());
        post_alias_columns(&gradient,&outputs[0],Some(&[&outputs[1]])).unwrap();
        assert_eq!(current_tangents(&outputs[0]).unwrap().unwrap()[0].id(),outputs[1].id());
        drop(frame);drop(level);
        close(&outputs[1].to_vec(),&[0.19968034,0.,0.,-0.3899324]);
        assert!(outputs[1].requires_grad());
        let query=outputs[1].sum().grad_wrt(&[&w],true).unwrap().remove(0);
        close(&query.to_vec(),&[2.*outputs[1].to_vec().iter().sum::<f32>()]);
        assert!(!query.requires_grad());
        assert!(x.grad().is_none()&&w.grad().is_none());
        assert!(is_grad_enabled()&&!is_active());
    }
}

#[test]
fn physical_batch_random_policy_uses_one_primal_draw_and_preserves_live_wide_seed() {
    let x=tensor(&[1.,2.],&[2]);
    let seed=Tensor::from_vec_f64(vec![1.,0.,0.,1.],&[2,2]).unwrap().requires_grad_(true).unwrap();
    let rng=ferro_core::Rng::new(7);let state=rng.state();
    let level=DualLevel::functional_batched(&[&x],&[&seed],2).unwrap();
    assert!(Tensor::rand_f64(&[2],&rng).is_err());
    assert!(Tensor::rand_f64(&[0],&rng).is_err());
    assert_eq!(rng.state(),state);
    assert!(x.dropout(0.5,true,7,0).is_err());
    let empty=Tensor::zeros(&[2,0]);assert_eq!(empty.dropout(0.5,true,7,0).unwrap().id(),empty.id());
    level.set_randomness_same(true).unwrap();
    let draw=Tensor::rand_f64(&[2],&rng).unwrap().to_dtype(DType::F32);
    let output=x.mul(&draw).unwrap();
    let tangent=level.tangents(&output).unwrap().unwrap().remove(0);
    assert_eq!(tangent.dtype(),DType::F64);
    drop(level);
    let control=ferro_core::Rng::new(7);let expected=Tensor::rand_f64(&[2],&control).unwrap();
    assert_eq!(rng.state(),control.state());assert_eq!(draw.to_vec(),expected.to_vec());
    let d=draw.to_vec_f64();assert_eq!(tangent.to_vec_f64(),vec![d[0],0.,0.,d[1]]);
    let gradient=tangent.sum().grad_wrt(&[&seed],false).unwrap().remove(0);
    assert_eq!(gradient.to_vec_f64(),vec![d[0],d[1],d[0],d[1]]);
    assert!(seed.grad().is_none()&&x.grad().is_none());
}


#[test]
fn physical_opaque_fallback_contracts_one_local_jacobian_without_replaying_the_primal() {
    let x=tensor(&[0.2,-0.4],&[2]);
    let seed=tensor(&[1.,2.,3.,4.,5.,6.],&[3,2]).requires_grad_(true).unwrap();
    let calls=Arc::new(AtomicUsize::new(0));let observed=calls.clone();
    let level=DualLevel::functional_batched(&[&x],&[&seed],3).unwrap();
    let output=Tensor::ones(&[2]).record_fn(vec![x.clone()],move|g|{observed.fetch_add(1,Ordering::Relaxed);vec![g.clone()]});
    let derivative=level.batched_tangent(&output).unwrap().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed),2);
    assert_eq!(derivative.to_vec(),seed.to_vec());
    drop(level);
    assert_eq!(derivative.sum().grad_wrt(&[&seed],false).unwrap().remove(0).to_vec(),vec![1.;6]);
    assert!(x.grad().is_none()&&seed.grad().is_none());
    for count in [0,2] {
        let x=Tensor::from_vec(vec![0.2;count],&[count]).unwrap();
        let direction=Tensor::ones(&[3,count]);
        let level=DualLevel::functional_batched(&[&x],&[&direction],3).unwrap();
        let output=x.gelu_erf();
        let derivative=level.batched_tangent(&output).unwrap().unwrap();
        assert_eq!(derivative.shape(),&[3,count]);
        for value in derivative.to_vec() {assert!((value-0.6574682).abs()<2e-6);}
        drop(level);
    }
}


#[test]
fn physical_clamp_uses_conditional_selection_for_nonfinite_directions() {
    let x=tensor(&[-2.,0.,2.],&[3]);
    let seed=tensor(&[f32::NAN,f32::NAN,f32::NAN],&[1,3]).requires_grad_(true).unwrap();
    let level=DualLevel::functional_batched(&[&x],&[&seed],1).unwrap();
    let output=x.clamp(-1.,1.);
    let tangent=level.batched_tangent(&output).unwrap().unwrap();
    let values=tangent.to_vec();assert_eq!(values[0],0.);assert!(values[1].is_nan());assert_eq!(values[2],0.);
    drop(level);
    assert_eq!(tangent.sum().grad_wrt(&[&seed],false).unwrap().remove(0).to_vec(),vec![0.,1.,0.]);
}
