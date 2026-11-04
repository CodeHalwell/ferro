use ferro_core::{Tensor,DType};
use ferro_core::forward_ad::DualLevel;

fn exact(actual:&[f32],expected:&[f32]) {
    for (&a,&b) in actual.iter().zip(expected) {assert!(a.is_nan()&&b.is_nan()||a.to_bits()==b.to_bits(),"{a:?} != {b:?}");}
}
fn power(x:f32,p:f32)->f32 {if p==0.5{x.sqrt()}else if p== -0.5{x.sqrt().recip()}else if p== -1.{x.recip()}else if p== -2.{(x*x).recip()}else if p==2.{x*x}else if p==3.{(x*x)*x}else{x.powf(p)}}

#[test]
fn half_power_primal_values_preserve_invalid_domains_and_signed_zero() {
    let x=Tensor::from_vec(vec![f32::NEG_INFINITY,-4.,-0.,0.,4.,f32::INFINITY,f32::NAN],&[7]).unwrap();
    exact(&x.powf(0.5).to_vec(),&[f32::NAN,f32::NAN,-0.,0.,2.,f32::INFINITY,f32::NAN]);
    exact(&x.powf(-0.5).to_vec(),&[f32::NAN,f32::NAN,f32::NEG_INFINITY,f32::INFINITY,0.5,0.,f32::NAN]);
    assert!(!x.powf(0.5).requires_grad());
}

#[test]
fn ordinary_and_graph_power_adjoints_stage_slope_before_the_cotangent() {
    let values=[f32::NEG_INFINITY,-1.,-0.,0.,1e-20,0.2,1.,2.,1e20,f32::INFINITY,f32::NAN];
    for p in [-3.,-2.,-1.5,-1.,-0.5,0.,0.5,1.,2.,3.] {for amount in [f32::from_bits(1),0.7,1e38] {for graph in [false,true] {
        let x=Tensor::from_vec(values.to_vec(),&[values.len()]).unwrap().requires_grad_(true).unwrap();
        let g=Tensor::full(x.shape(),amount).requires_grad_(true).unwrap();let y=x.powf(p);
        let result=Tensor::vjp_many_wrt_optional(&[&y],&[&x],&[&g],graph).unwrap().remove(0).unwrap();
        let expected=values.iter().map(|&x|if p==0.{0.}else{amount*(p*power(x,p-1.))}).collect::<Vec<_>>();
        exact(&result.to_vec(),&expected);assert_eq!(result.requires_grad(),graph&&p!=0.);
        assert!(x.grad().is_none()&&g.grad().is_none());
        if p==1. {assert_ne!(result.id(),g.id());}
    }}}
    let x=Tensor::from_vec(vec![1e-10],&[1]).unwrap().requires_grad_(true).unwrap();
    let g=Tensor::full(&[1],2e38);let y=x.powf(3.);
    let result=y.vjp_wrt(&[&x],&g,false).unwrap().remove(0);
    assert!(result.item().is_finite());assert!((2e38f32*3.0).is_infinite());
}

#[test]
fn power_forward_directions_keep_primal_slope_precision_and_zero_output_dtype() {
    for p in [-0.5,0.,0.5,1.,3.] {
        let x=Tensor::from_vec(vec![0.2,0.7,2.],&[3]).unwrap().requires_grad_(true).unwrap();
        let v=Tensor::from_vec_f64(vec![0.3,0.7,1.1],&[3]).unwrap().requires_grad_(true).unwrap();
        let level=DualLevel::new(&[&x],&[&v]).unwrap();let y=x.powf(p);let tangent=level.tangent(&y).unwrap().unwrap();drop(level);
        assert_eq!(tangent.dtype(),if p==0.{DType::F32}else{DType::F64});assert_eq!(tangent.requires_grad(),p!=0.);
        let expected=x.to_vec().into_iter().zip(v.to_vec_f64()).map(|(x,v)|if p==0.{0.}else{v*(p*power(x,p-1.))as f64}).collect::<Vec<_>>();
        assert_eq!(tangent.to_vec_f64(),expected);
    }
}

#[test]
fn power_guards_input_values_even_when_the_slope_is_zero_or_identity() {
    for p in [0.,0.5,1.,-0.5] {
        let x=Tensor::from_vec(vec![0.2,0.7],&[2]).unwrap().requires_grad_(true).unwrap();let y=x.powf(p);
        y.detach().fill_(0.).unwrap();assert!(y.vjp_wrt(&[&x],&Tensor::ones(&[2]),false).is_ok());
        x.detach().fill_(3.).unwrap();assert!(y.vjp_wrt(&[&x],&Tensor::ones(&[2]),false).is_err());assert!(x.grad().is_none());
    }
}
