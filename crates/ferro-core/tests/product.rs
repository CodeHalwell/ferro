use ferro_core::{DType, Tensor};
use ferro_core::autograd::{is_grad_enabled, with_grad_enabled};
use ferro_core::forward_ad::DualLevel;

fn t(values: &[f64], shape: &[usize], dtype: DType, tracked: bool) -> Tensor {
    Tensor::from_vec_f64(values.to_vec(), shape).unwrap().to_dtype(dtype).requires_grad_(tracked).unwrap()
}
fn close(actual: &Tensor, expected: &[f64]) {
    assert_eq!(actual.numel(), expected.len());
    for (a,b) in actual.to_vec_f64().into_iter().zip(expected) { assert!(a.is_nan() && b.is_nan() || a == *b || (a-b).abs() <= 2e-6*b.abs().max(1.), "{a} != {b}"); }
}

#[test]
fn typed_values_strides_scalars_empty_and_wrapping() {
    for dtype in [DType::F32,DType::F64] {
        let x=t(&[2.,3.,4.,5.,6.,7.],&[2,3],dtype,false).transpose(0,1).unwrap();
        close(&x.product(Some(1),false,None).unwrap(),&[10.,18.,28.]);
        assert_eq!(x.product(Some(0),true,None).unwrap().shape(),&[1,2]);
        close(&x.product(None,false,None).unwrap(),&[5040.]);
        close(&t(&[3.],&[],dtype,false).product(Some(0),true,None).unwrap(),&[3.]);
        let empty=Tensor::zeros(&[2,0,3]).to_dtype(dtype);
        close(&empty.product(Some(1),false,None).unwrap(),&[1.;6]);
        close(&empty.product(None,false,None).unwrap(),&[1.]);
    }
    let b=Tensor::from_vec_bool(vec![true,true,false,true],&[2,2]).unwrap();
    let out=b.product(Some(1),false,None).unwrap();assert_eq!(out.dtype(),DType::I64);assert_eq!(out.to_vec_i64(),vec![1,0]);
    assert_eq!(b.product(Some(0),false,Some(DType::Bool)).unwrap().to_vec_bool(),vec![false,true]);
    let integer=Tensor::from_vec_i64(vec![i64::MAX,2,-1],&[3]).unwrap();
    assert_eq!(integer.product(None,false,None).unwrap().to_vec_i64(),vec![2]);
    let x=t(&[2.2,3.1],&[2],DType::F32,true);
    assert!(!x.product(None,false,Some(DType::I64)).unwrap().requires_grad());
}

#[test]
fn ordinary_zero_nonfinite_and_consumed_dtype_staging() {
    for dtype in [DType::F32,DType::F64] {
        let x=t(&[0.,f64::INFINITY,0.],&[3],dtype,true);
        let all=x.product(None,false,None).unwrap();let dim=x.product(Some(0),false,None).unwrap();
        close(&all.grad_wrt(&[&x],false).unwrap()[0],&[0.,0.,0.]);
        close(&dim.grad_wrt(&[&x],false).unwrap()[0],&[f64::NAN,0.,f64::NAN]);
        let nonzero=t(&[f64::INFINITY],&[1],dtype,true);
        close(&nonzero.product(None,false,None).unwrap().grad_wrt(&[&nonzero],false).unwrap()[0],&[f64::NAN]);
    }
    let x=t(&[1e-50,2.],&[2],DType::F64,true);
    close(&x.product(None,false,Some(DType::F32)).unwrap().grad_wrt(&[&x],false).unwrap()[0],&[2.,0.]);
    let x=t(&[1e30,1e30,1e-30],&[3],DType::F32,false);
    assert!(x.product(None,false,None).unwrap().item().is_infinite());
    assert!(x.cumprod(0).unwrap().to_vec()[2].is_finite());
}

#[test]
fn finite_difference_and_live_second_third_cotangent_derivatives() {
    let x=t(&[2.,3.,4.],&[3],DType::F32,true);
    let actual=x.product(Some(0),false,None).unwrap().grad_wrt(&[&x],false).unwrap().remove(0);
    for axis in 0..3 {
        let mut plus=vec![2.,3.,4.];let mut minus=plus.clone();plus[axis]+=1e-3;minus[axis]-=1e-3;
        let fd=(t(&plus,&[3],DType::F32,false).product(None,false,None).unwrap().item()-t(&minus,&[3],DType::F32,false).product(None,false,None).unwrap().item())/2e-3;
        assert!((actual.to_vec()[axis]-fd).abs()<0.003);
    }
    let x=t(&[2.,0.,4.],&[3],DType::F32,true);let seed=t(&[3.],&[],DType::F32,true);
    let first=x.product(None,false,None).unwrap().vjp_wrt(&[&x],&seed,true).unwrap().remove(0);
    close(&first,&[0.,24.,0.]);
    let second=first.sum().grad_wrt(&[&x,&seed],true).unwrap();
    close(&second[0],&[12.,18.,6.]);close(&second[1],&[8.]);
    close(&second[0].sum().grad_wrt(&[&x],false).unwrap()[0],&[6.,6.,6.]);
    assert!(x.grad().is_none()&&seed.grad().is_none());
}

#[test]
fn functional_query_policy_is_scoped_and_returned_primal_stays_ordinary() {
    let x=t(&[f64::INFINITY],&[1],DType::F32,true);let y=x.product(None,false,None).unwrap();let seed=Tensor::scalar(1.);
    for enabled in [false,true] {
        let got=with_grad_enabled(enabled,||Tensor::vjp_many_wrt_func(&[&y],&[&x],&[&seed],false,true,None)).unwrap();
        close(got[0].as_ref().unwrap(),&[1.]);assert!(is_grad_enabled());
        assert!(y.grad_wrt(&[&x],false).unwrap()[0].to_vec()[0].is_nan());
    }
    assert!(Tensor::vjp_many_wrt_func(&[&y],&[&x],&[],false,true,None).is_err());
    assert!(y.grad_wrt(&[&x],false).unwrap()[0].to_vec()[0].is_nan());
    let bomb=Tensor::scalar(1.).record_fn(vec![x.clone()], |_|panic!("deliberate callback"));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(||Tensor::vjp_many_wrt_func(&[&bomb],&[&x],&[&seed],false,true,None))).is_err());
    assert!(y.grad_wrt(&[&x],false).unwrap()[0].to_vec()[0].is_nan());
}

#[test]
fn explicit_query_policy_restores_after_nested_ordinary_and_functional_queries() {
    let input=Tensor::scalar(2.).requires_grad_(true).unwrap();
    let held=t(&[f64::INFINITY],&[1],DType::F32,true);
    let op=held.product(None,false,None).unwrap();let op_clone=op.clone();let held_clone=held.clone();
    let root=Tensor::scalar(3.).record_fn(vec![input.clone()],move |g| {
        let seed=Tensor::scalar(1.);
        assert!(op_clone.grad_wrt(&[&held_clone],false).unwrap()[0].to_vec()[0].is_nan());
        let functional=Tensor::vjp_many_wrt_func(&[&op_clone],&[&held_clone],&[&seed],false,true,None).unwrap();
        close(functional[0].as_ref().unwrap(),&[1.]);
        assert!(op_clone.grad_wrt(&[&held_clone],false).unwrap()[0].to_vec()[0].is_nan());
        vec![g.clone()]
    });
    let seed=Tensor::scalar(1.);
    close(Tensor::vjp_many_wrt_func(&[&root],&[&input],&[&seed],false,true,None).unwrap()[0].as_ref().unwrap(),&[1.]);
    assert!(op.grad_wrt(&[&held],false).unwrap()[0].to_vec()[0].is_nan());
    assert!(input.grad().is_none()&&held.grad().is_none());
}

#[test]
fn forward_functional_mode_changes_coefficients_without_changing_primal() {
    for functional in [false,true] {
        let x=t(&[f64::INFINITY],&[1],DType::F32,true);let seed=t(&[1.],&[1],DType::F64,true);
        let level=if functional { DualLevel::functional(&[&x],&[&seed]) } else { DualLevel::new(&[&x],&[&seed]) }.unwrap();
        let y=x.product(None,false,None).unwrap();let tangent=level.tangent(&y).unwrap().unwrap();drop(level);
        assert_eq!(tangent.dtype(),DType::F64);
        close(&tangent,&[if functional {1.}else{f64::NAN}]);
        assert!(y.grad_wrt(&[&x],false).unwrap()[0].to_vec()[0].is_nan());
    }
    let x=t(&[2.,0.,4.],&[3],DType::F32,true);let seed=t(&[0.1,0.2,0.3],&[3],DType::F32,true);
    let level=DualLevel::functional(&[&x],&[&seed]).unwrap();let y=x.product(None,false,None).unwrap();let tangent=level.tangent(&y).unwrap().unwrap();drop(level);
    close(&tangent,&[1.6]);let gradients=tangent.grad_wrt(&[&x,&seed],false).unwrap();
    close(&gradients[0],&[0.8,1.,0.4]);close(&gradients[1],&[0.,8.,0.]);
}

#[test]
fn saved_inputs_outputs_and_functional_tangent_versions_are_distinct() {
    for output in [false,true] {
        let x=t(&[2.,3.],&[2],DType::F32,true);let y=x.product(None,false,None).unwrap();
        with_grad_enabled(false,||if output{y.detach().fill_(9.)}else{x.fill_(9.)}).unwrap();
        assert!(y.grad_wrt(&[&x],false).is_err());assert!(x.grad().is_none());
    }
    for changed in 0..3 {
        let x=t(&[2.,3.],&[2],DType::F32,true);let seed=t(&[1.,1.],&[2],DType::F32,true);
        let level=DualLevel::functional(&[&x],&[&seed]).unwrap();let y=x.product(None,false,None).unwrap();let tangent=level.tangent(&y).unwrap().unwrap();drop(level);
        with_grad_enabled(false,||[&x,&y,&seed][changed].detach().fill_(9.)).unwrap();
        let result=tangent.grad_wrt(&[&x,&seed],false);
        if changed==2{assert!(result.is_err());}else{let result=result.unwrap();close(&result[0],&[1.,1.]);close(&result[1],&[3.,2.]);}
    }
}

#[test]
fn zero_size_and_derived_output_capacity_are_checked_before_work() {
    let huge=Tensor::zeros(&[isize::MAX as usize/8+1,0]);
    assert!(huge.product(Some(1),false,None).is_err());
    assert_eq!(huge.product(Some(0),false,None).unwrap().shape(),&[0]);
    close(&huge.product(None,false,None).unwrap(),&[1.]);
    let x=Tensor::scalar(1.).expand(&[isize::MAX as usize/8+1]).unwrap();
    assert!(x.product(None,false,None).is_err());
    let x=Tensor::ones(&[2,3]);assert!(x.product(Some(2),false,None).is_err());assert!(x.product(None,true,None).is_err());
    assert!(ferro_core::capture(||x.product(None,false,None)).is_err());
}

#[test]
fn caller_recording_policy_is_independent_of_graph_construction_and_restores() {
    let x=t(&[0.,f64::INFINITY,0.],&[3],DType::F32,true);
    let y=x.product(None,false,None).unwrap();let seed=Tensor::scalar(1.);
    for caller in [false,true] {
        let actual=Tensor::vjp_many_wrt_transform_policy(&[&y],&[&x],&[&seed],false,true,None,false,caller).unwrap().remove(0).unwrap();
        close(&actual,if caller { &[f64::NAN,0.,f64::NAN] } else { &[0.,0.,0.] });
        assert!(!actual.requires_grad());
        close(&y.grad_wrt(&[&x],false).unwrap()[0],&[0.,0.,0.]);
    }
    let held=x.clone();let product=y.clone();
    let root=Tensor::scalar(2.).record_fn(vec![x.clone()],move |_| {
        close(&product.grad_wrt(&[&held],false).unwrap()[0],&[0.,0.,0.]);
        let one=Tensor::scalar(1.);
        let restored=Tensor::vjp_many_wrt_transform_policy(&[&product],&[&held],&[&one],false,true,None,false,true).unwrap();
        close(restored[0].as_ref().unwrap(),&[f64::NAN,0.,f64::NAN]);
        vec![Tensor::zeros(&[3])]
    });
    Tensor::vjp_many_wrt_transform_policy(&[&root],&[&x],&[&seed],false,true,None,false,true).unwrap();
    let bomb=Tensor::scalar(1.).record_fn(vec![x.clone()],|_|panic!("deliberate caller-mode unwind"));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(||Tensor::vjp_many_wrt_transform_policy(&[&bomb],&[&x],&[&seed],false,true,None,false,true))).is_err());
    assert!(Tensor::vjp_many_wrt_transform_policy(&[&y],&[&x],&[],false,true,None,false,true).is_err());
    close(&y.grad_wrt(&[&x],false).unwrap()[0],&[0.,0.,0.]);
    y.backward();close(&x.grad().unwrap(),&[0.,0.,0.]);
    assert!(is_grad_enabled());
}
