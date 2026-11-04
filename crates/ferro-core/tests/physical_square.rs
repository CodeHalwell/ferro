use ferro_core::{Tensor,DType};
use ferro_core::forward_ad::{DualLevel,is_active};
struct Entry { values:&'static[f64],live:bool,marker:bool }
struct Golden { physical:bool,wide:bool,input:f64,seed:f64,tangent:&'static[f64],first:[Entry;2],second:Option<[Entry;2]> }
include!("fixtures/physical2_square_goldens.rs");
fn exact(values:&[f64],expected:&[f64],context:&str){
    assert_eq!(values.len(),expected.len(),"{context}");
    for(i,(&a,&b))in values.iter().zip(expected).enumerate(){
        assert!(a.to_bits()==b.to_bits()||(a.is_nan()&&b.is_nan()),"{context}[{i}]: {a} != {b}");
    }
}
#[test]
fn power_provenance_and_coefficient_first_rounding_match_actual_torch(){
    for(i,golden)in GOLDENS.iter().enumerate(){
        let x=Tensor::from_vec(vec![golden.input as f32],&[1]).unwrap().requires_grad_(true).unwrap();
        let raw=if golden.physical{vec![golden.seed,0.7,-0.3]}else{vec![golden.seed]};
        let shape=if golden.physical{vec![3,1]}else{vec![1]};
        let v=if golden.wide{Tensor::from_vec_f64(raw,&shape).unwrap()}else{Tensor::from_vec(raw.iter().map(|&x|x as f32).collect(),&shape).unwrap()}.requires_grad_(true).unwrap();
        let level=if golden.physical{DualLevel::functional_batched(&[&x],&[&v],3)}else{DualLevel::functional(&[&x],&[&v])}.unwrap();
        let output=x.square().unwrap();let tangent=level.tangents(&output).unwrap().unwrap().remove(0);drop(level);
        exact(&tangent.to_vec_f64(),golden.tangent,&format!("case{i} tangent"));
        assert_eq!(tangent.dtype(),if golden.wide{DType::F64}else{DType::F32});
        assert!(tangent.requires_grad());
        let first=tangent.sum().grad_wrt(&[&x,&v],!golden.wide).unwrap();
        for(slot,(value,expected))in first.iter().zip(&golden.first).enumerate(){
            exact(&value.to_vec_f64(),expected.values,&format!("case{i} first{slot}"));
            assert_eq!(value.requires_grad(),expected.live,"case{i} first{slot} flag");
            assert_eq!(value.is_structural_zero(),expected.marker,"case{i} first{slot} marker");
        }
        if let Some(expected)=&golden.second{
            let second=first[0].sum().vjp_wrt_optional(&[&x,&v],&Tensor::scalar(1.),true).unwrap();
            for(slot,(value,expected))in second.iter().zip(expected).enumerate(){
                let value=value.as_ref().expect("recorded derivative remains connected");
                exact(&value.to_vec_f64(),expected.values,&format!("case{i} second{slot}"));
                assert_eq!(value.requires_grad(),expected.live,"case{i} second{slot} flag");
                assert_eq!(value.is_structural_zero(),expected.marker,"case{i} second{slot} marker");
            }
        }
        assert!(x.grad().is_none()&&v.grad().is_none()&&!is_active());
    }
    println!("{} Torch Square controls passed",GOLDENS.len());
}

#[test]
fn square_graph_backward_retains_power_history_and_empty_shapes(){
    for shape in [&[][..],&[2][..],&[0,3][..]]{
        let x=Tensor::full(shape,2.).requires_grad_(true).unwrap();
        let first=x.square().unwrap().sum().grad_wrt(&[&x],true).unwrap().remove(0);
        let second=first.sum().grad_wrt(&[&x],true).unwrap().remove(0);
        assert!(second.requires_grad());assert!(second.to_vec().iter().all(|&x|x==2.));
        let third=second.sum().grad_wrt(&[&x],true).unwrap().remove(0);
        assert!(!third.requires_grad()&&!third.is_structural_zero());assert!(third.to_vec().iter().all(|&x|x==0.));
    }
}
