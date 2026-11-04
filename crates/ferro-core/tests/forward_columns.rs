use ferro_core::{Tensor, DType};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::{DualLevel, ReverseFrameGuard, current_tangents, prepare_alias, post_alias_columns, unwrap_values, is_active};
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

fn tensor(data: &[f32]) -> Tensor { Tensor::from_vec(data.to_vec(), &[data.len()]).unwrap() }
fn leaf(data: &[f32]) -> Tensor { tensor(data).requires_grad_(true).unwrap() }
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a,b) in actual.iter().zip(expected) { assert!((a-b).abs()<5e-6, "{a} != {b}"); }
}

#[test]
fn independent_columns_share_one_primal_and_keep_each_live_direction_graph() {
    let x = leaf(&[0.2,-0.4]);
    let a = leaf(&[0.3,0.7]);
    let b = leaf(&[-0.2,0.6]);
    let w = leaf(&[0.5,0.8]);
    let level = DualLevel::functional_columns(&[&x], &[&[&a], &[&b]]).unwrap();
    assert!(level.tangent(&x).is_err());
    assert_eq!(level.tangents(&x).unwrap().unwrap().iter().map(Tensor::id).collect::<Vec<_>>(), vec![a.id(),b.id()]);
    let calls = AtomicUsize::new(0);
    let function = || { calls.fetch_add(1, Ordering::Relaxed); x.mul(&x).unwrap().mul(&w).unwrap() };
    let y = function();
    let values = level.tangents(&y).unwrap().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed),1);
    drop(level);
    close(&values[0].to_vec(), &[0.06,-0.448]);
    close(&values[1].to_vec(), &[-0.04,-0.384]);
    for (direction, seed) in values.iter().zip([&a,&b]) {
        let grads=direction.sum().grad_wrt(&[&x,seed,&w],true).unwrap();
        close(&grads[0].to_vec(), &seed.to_vec().iter().zip(w.to_vec()).map(|(v,w)|2.*v*w).collect::<Vec<_>>());
        close(&grads[1].to_vec(), &[0.2,-0.64]);
        close(&grads[2].to_vec(), &x.to_vec().iter().zip(seed.to_vec()).map(|(x,v)|2.*x*v).collect::<Vec<_>>());
    }
    assert!([&x,&a,&b,&w].iter().all(|x|x.grad().is_none()));
}

#[test]
fn columns_cross_no_grad_islands_and_private_reverse_frame_without_synthetic_edges() {
    for input_grad in [false,true] {
        let x=tensor(&[0.2,-0.4]).requires_grad_(input_grad).unwrap();
        let w=Tensor::scalar(0.5).requires_grad_(true).unwrap();
        let a=tensor(&[1.,0.]); let b=tensor(&[0.,1.]);
        let level=DualLevel::functional_columns(&[&x], &[&[&a], &[&b]]).unwrap();
        let frame=ReverseFrameGuard::new().unwrap();
        let root=if input_grad { x.reshape(&[2]).unwrap() } else { x.detach().requires_grad_(true).unwrap() };
        prepare_alias(&x,&root).unwrap();
        let coefficient=with_grad_enabled(false, || root.mul(&root).unwrap().tanh());
        let loss=root.mul(&coefficient).unwrap().mul(&w).unwrap().sum();
        let gradient=loss.vjp_wrt_forward(&[&root],&Tensor::scalar(1.)).unwrap().remove(0).unwrap();
        let columns=current_tangents(&gradient).unwrap().unwrap();
        let outputs=unwrap_values(&[&gradient,&columns[0],&columns[1]]).unwrap();
        post_alias_columns(&gradient,&outputs[0],Some(&[&outputs[1],&outputs[2]])).unwrap();
        assert_eq!(current_tangents(&outputs[0]).unwrap().unwrap()[1].id(),outputs[2].id());
        drop(frame); drop(level);
        close(&outputs[1].to_vec(),&[0.19968034,0.]);
        close(&outputs[2].to_vec(),&[0.,-0.3899324]);
        for column in &outputs[1..] {
            assert!(column.requires_grad());
            let query=column.sum().grad_wrt(&[&w],true).unwrap();
            close(&query[0].to_vec(),&[2.*column.to_vec().iter().sum::<f32>()]);
            assert!(!query[0].requires_grad());
        }
        assert!(x.grad().is_none() && w.grad().is_none());
    }
}

#[test]
fn validation_and_late_column_failures_never_publish_partial_tangents() {
    let x=tensor(&[1.,2.]); let v=tensor(&[1.,0.]); let bad=Tensor::scalar(1.);
    assert!(DualLevel::functional_columns(&[&x],&[]).is_err());
    assert!(DualLevel::functional_columns(&[&x],&[&[]]).is_err());
    assert!(DualLevel::functional_columns(&[&x],&[&[&v],&[&bad]]).is_err());
    assert!(!is_active());
    let level=DualLevel::functional_columns(&[&x],&[&[&v],&[&v]]).unwrap();
    assert!(DualLevel::new(&[&x],&[&v]).is_err());
    let calls=Arc::new(AtomicUsize::new(0)); let seen=calls.clone();
    let out=Tensor::scalar(0.).record_fn(vec![x.clone()],move |_| {
        let index=seen.fetch_add(1,Ordering::Relaxed);
        vec![if index==0 {Tensor::ones(&[2])} else {Tensor::scalar(0.)}]
    });
    assert!(level.tangents(&out).is_err());
    assert!(level.tangents(&x).is_err());
    assert_eq!(calls.load(Ordering::Relaxed),2);
    assert!(x.grad().is_none() && out.grad().is_none());
    drop(level);
    assert!(!is_active());
    let level=DualLevel::functional_columns(&[&x],&[&[&v],&[&v]]).unwrap();
    assert_eq!(level.tangents(&x).unwrap().unwrap().len(),2);
}

#[test]
fn mixed_direction_dtypes_and_huge_empty_views_keep_column_metadata() {
    let x=tensor(&[0.2,0.4]); let a=tensor(&[1.,2.]);
    let b=Tensor::from_vec_f64(vec![0.3,0.7],&[2]).unwrap();
    let level=DualLevel::functional_columns(&[&x],&[&[&a],&[&b]]).unwrap();
    let out=x.unsqueeze(0).unwrap().expand(&[2,2]).unwrap();
    let columns=level.tangents(&out).unwrap().unwrap();
    assert_eq!(columns[0].dtype(),DType::F32); assert_eq!(columns[1].dtype(),DType::F64);
    assert_eq!(columns[1].to_vec_f64(),vec![0.3,0.7,0.3,0.7]);
    assert_eq!(columns[0].strides(),vec![0,1]);
    drop(level);
    let x=Tensor::zeros(&[0,usize::MAX]); let a=x.detach(); let b=x.detach();
    let level=DualLevel::functional_columns(&[&x],&[&[&a],&[&b]]).unwrap();
    let out=x.permute(&[1,0]).unwrap();
    let columns=level.tangents(&out).unwrap().unwrap();
    assert!(columns.iter().all(|column| column.shape()==[usize::MAX,0] && column.to_vec().is_empty()));
}

#[test]
fn postprocessing_checks_all_columns_before_changing_alias_map() {
    let x=tensor(&[1.,2.]); let a=tensor(&[1.,0.]); let b=tensor(&[0.,1.]);
    let level=DualLevel::functional_columns(&[&x],&[&[&a],&[&b]]).unwrap();
    let alias=x.detach();
    assert!(post_alias_columns(&x,&alias,Some(&[&a])).is_err());
    assert!(level.tangents(&alias).unwrap().is_none());
    assert!(post_alias_columns(&x,&alias,Some(&[&a,&Tensor::scalar(1.)])).is_err());
    assert!(level.tangents(&alias).unwrap().is_none());
    post_alias_columns(&x,&alias,Some(&[&a,&b])).unwrap();
    assert_eq!(level.tangents(&alias).unwrap().unwrap()[1].id(),b.id());
    post_alias_columns(&x,&alias,None).unwrap();
    assert!(level.tangents(&alias).unwrap().is_none());
}

#[test]
fn random_error_policy_rejects_before_generator_draw_and_same_uses_one_primal() {
    let x=tensor(&[1.,2.]); let a=tensor(&[1.,0.]); let b=tensor(&[0.,1.]);
    let rng=ferro_core::Rng::new(7); let state=rng.state();
    let level=DualLevel::functional_columns(&[&x],&[&[&a],&[&b]]).unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Tensor::randn(&[2],&rng))).is_err());
    assert!(Tensor::rand_f64(&[2],&rng).is_err());
    assert!(Tensor::randn_f64(&[0],&rng).is_err());
    assert_eq!(rng.state(),state);
    assert!(x.dropout(0.5,true,7,0).is_err());
    let empty=Tensor::zeros(&[2,0]);
    assert_eq!(empty.dropout(0.5,true,7,0).unwrap().id(),empty.id());
    let empty=Tensor::from_vec_f64(vec![],&[2,0]).unwrap();
    assert_eq!(empty.dropout_f64(0.5,true,7,0).unwrap().id(),empty.id());
    assert_eq!(x.dropout(0.,true,7,0).unwrap().id(),x.id());
    // Local fallback callbacks run with tangent dispatch suspended; their
    // random operations must still obey the active transform's policy.
    let local=Tensor::scalar(0.).record_fn(vec![x.clone()], |_| {
        assert!(ferro_core::forward_ad::check_random_operation().is_err());
        vec![Tensor::ones(&[2])]
    });
    assert_eq!(level.tangents(&local).unwrap().unwrap().len(),2);
    level.set_randomness_same(true).unwrap();
    let draw=Tensor::rand_f64(&[2],&rng).unwrap().to_dtype(DType::F32);
    let out=x.mul(&draw).unwrap();
    let columns=level.tangents(&out).unwrap().unwrap();
    close(&columns[0].to_vec(),&[draw.to_vec()[0],0.]);
    close(&columns[1].to_vec(),&[0.,draw.to_vec()[1]]);
    let control=ferro_core::Rng::new(7);
    let expected=Tensor::rand_f64(&[2],&control).unwrap();
    assert_eq!(rng.state(),control.state());
    assert_eq!(draw.to_vec(),expected.to_vec());
    drop(level);
    assert!(Tensor::rand_f64(&[1],&rng).is_ok());
}
