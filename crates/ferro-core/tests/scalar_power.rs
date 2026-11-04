use ferro_core::{Tensor,DType};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::capture::capture;
use ferro_core::forward_ad::DualLevel;
#[path="fixtures/scalar_power_reference.rs"] mod reference;

fn close(a:f32,b:f32) {
    if a.is_nan()&&b.is_nan(){return;}
    if !a.is_finite()||!b.is_finite()||a==0.||b==0. {assert_eq!(a.to_bits(),b.to_bits(),"{a:?} != {b:?}");return;}
    assert!((a-b).abs() <= 4.*f32::EPSILON*b.abs(),"{a:?} != {b:?}");
}

#[test]
fn original_scalar_domain_and_staged_adjoint_match_torch_scalar_goldens() {
    for &(p,x,y,dx,live) in reference::ROWS {
        let p=f64::from_bits(p);let x=Tensor::from_vec(vec![f32::from_bits(x)],&[1]).unwrap().requires_grad_(true).unwrap();
        let g=Tensor::full(&[1],0.7).requires_grad_(true).unwrap();let out=x.pow_scalar(p).unwrap();
        close(out.item(),f32::from_bits(y));
        for graph in [false,true] {
            let actual=out.vjp_wrt(&[&x],&g,graph).unwrap().remove(0);
            close(actual.item(),f32::from_bits(dx));assert_eq!(actual.requires_grad(),graph&&live,"power={p}");
        }
        assert!(x.grad().is_none()&&g.grad().is_none());
    }
}

#[test]
fn portable_scalar_domain_is_not_changed_by_tensor_length_or_special_power_narrowing() {
    for p in [f64::from_bits(0.5f64.to_bits()+1),f64::from_bits((-0.5f64).to_bits()-1),f64::from_bits(1f64.to_bits()+1),f64::from_bits(2f64.to_bits()+1),f64::from_bits(1)] {
        for count in [1,5,32,33,1000] {
            let values=[-4f32,-0.,0.,4.,f32::NEG_INFINITY].into_iter().cycle().take(count).collect::<Vec<_>>();
            let x=Tensor::from_vec(values.clone(),&[count]).unwrap();
            let y=x.pow_scalar(p).unwrap();
            for (a,b) in y.to_vec().into_iter().zip(values) {close(a,(b as f64).powf(p) as f32);}
        }
    }
    let zeros=Tensor::from_vec(vec![-0.,0.],&[2]).unwrap();
    assert_eq!(zeros.pow_scalar(0.5).unwrap().to_vec()[0].to_bits(),(-0f32).to_bits());
    assert_eq!(zeros.pow_scalar(f64::from_bits(0.5f64.to_bits()+1)).unwrap().to_vec()[0].to_bits(),0f32.to_bits());
}

#[test]
fn original_scalar_graph_direction_layout_and_saved_versions_remain_real() {
    let x=Tensor::from_vec(vec![0.2,0.7,1.3,2.,3.,4.],&[2,3]).unwrap().transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
    for power in [1.7, f64::from_bits(1f64.to_bits()+1),0.,1.] {
        let y=x.pow_scalar(power).unwrap();assert_eq!(y.strides(),x.strides());assert_ne!(y._storage_ptr(),x._storage_ptr());
        let seed=Tensor::scalar(0.7).expand(y.shape()).unwrap().requires_grad_(true).unwrap();
        let first=y.vjp_wrt(&[&x],&seed,true).unwrap().remove(0);
        assert_eq!(first.requires_grad(),power!=0.);assert_ne!(first.id(),seed.id());
        if power!=0. {
            let second=first.sum().grad_wrt(&[&x],true).unwrap().remove(0);
            assert_eq!(second.requires_grad(),power!=1.);
        }
        let v=Tensor::from_vec_f64(vec![0.3,0.7,1.1,0.2,-0.4,0.8],&[2,3]).unwrap().transpose(0,1).unwrap().requires_grad_(true).unwrap();
        let level=DualLevel::new(&[&x],&[&v]).unwrap();let output=x.pow_scalar(power).unwrap();let tangent=level.tangent(&output).unwrap().unwrap();drop(level);
        assert_eq!(tangent.dtype(),if power==0.{DType::F32}else{DType::F64});
        if power!=0. {
            let slope=x.pow_scalar(power-1.).unwrap().mul(&Tensor::scalar(power as f32)).unwrap();
            assert_eq!(tangent.to_vec_f64(),v.to_vec_f64().into_iter().zip(slope.to_vec()).map(|(g,d)|g*d as f64).collect::<Vec<_>>());
            assert_eq!(tangent.strides(),v.strides());
        }
        let z=x.pow_scalar(power).unwrap();z.detach().transpose(0,1).unwrap().fill_(0.).unwrap();assert!(z.sum().grad_wrt(&[&x],false).is_ok());
    }
    let y=x.pow_scalar(1.7).unwrap();x.detach().transpose(0,1).unwrap().fill_(3.).unwrap();assert!(y.vjp_wrt(&[&x],&Tensor::ones(y.shape()),true).is_err());
    assert!(x.grad().is_none());
    assert!(Tensor::from_vec_i64(vec![1],&[1]).unwrap().pow_scalar(1.7).is_err());
}

#[test]
fn scalar_power_capture_replays_current_values_and_preserves_empty_metadata() {
    let x=Tensor::from_vec(vec![0.2,0.7,1.3,2.,3.,4.],&[2,3]).unwrap();
    for p in [1.7,f64::from_bits(0.5f64.to_bits()+1),0.5] {
        let y=capture(||x.transpose(0,1)?.pow_scalar(p)).unwrap();let chain=ferro_core::graph::CompiledChain::compile(&y).unwrap();
        x.fill_(2.).unwrap();let actual=chain.replay().unwrap();assert_eq!(actual.strides(),&[1,3]);
        for value in actual.to_vec(){close(value,2f64.powf(p) as f32);}
    }
    let empty=Tensor::zeros(&[0,usize::MAX]);let y=empty.pow_scalar(1.7).unwrap();assert_eq!(y.shape(),empty.shape());
    let huge=Tensor::scalar(1.).expand(&[isize::MAX as usize/4+1]).unwrap();assert!(huge.pow_scalar(1.7).is_err());
    let x=Tensor::ones(&[2]).requires_grad_(true).unwrap();let y=with_grad_enabled(false,||x.pow_scalar(1.7)).unwrap();assert!(!y.requires_grad());
}

#[path="fixtures/scalar_power_higher_reference.rs"] mod higher_reference;
#[test]
fn original_exponent_is_retained_through_repeated_derivative_power_history() {
    for &(p,value,second,second_live,third,third_live) in higher_reference::ROWS {
        let p=f64::from_bits(p);let x=Tensor::from_vec(vec![f32::from_bits(value)],&[1]).unwrap().requires_grad_(true).unwrap();let seed=Tensor::full(&[1],0.7).requires_grad_(true).unwrap();
        let y=x.pow_scalar(p).unwrap();let first=y.vjp_wrt(&[&x],&seed,true).unwrap().remove(0);
        let actual=first.grad_wrt(&[&x],true).unwrap().remove(0);close(actual.item(),f32::from_bits(second));assert_eq!(actual.requires_grad(),second_live,"{p}");
        if let Some(third)=third {let next=actual.grad_wrt(&[&x],true).unwrap().remove(0);close(next.item(),f32::from_bits(third));assert_eq!(next.requires_grad(),third_live,"{p}");}
        assert!(x.grad().is_none()&&seed.grad().is_none());
    }
}
