use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::{DualLevel, is_active};

struct Entry { values: &'static [f64], shape: &'static [usize], stride: &'static [usize], wide: bool, live: bool, marker: bool }
struct Golden {
    kind: &'static str, base: &'static [f64], shape: &'static [usize], transpose: bool,
    weights: &'static [f64], weight_shape: &'static [usize], dim: i32, keep: bool,
    outwide: bool, mask: usize, wide: bool, enabled: bool, seed_x: &'static [f64], seed_w: &'static [f64],
    output: Entry, next: &'static [Option<Entry>],
}
include!("fixtures/physical_current_compat_goldens.rs");

fn tensor(values: &[f64], shape: &[usize], wide: bool) -> Tensor {
    (if wide { Tensor::from_vec_f64(values.to_vec(),shape) }
     else { Tensor::from_vec(values.iter().map(|&value|value as f32).collect(),shape) }).unwrap().requires_grad_(true).unwrap()
}
fn check(value: &Tensor, expected: &Entry, label: &str) {
    assert_eq!(value.shape(),expected.shape,"{label} shape");
    assert_eq!(value.strides(),expected.stride,"{label} stride");
    assert_eq!(value.dtype(),if expected.wide {DType::F64}else{DType::F32},"{label} dtype");
    assert_eq!(value.requires_grad(),expected.live,"{label} recording flag");
    assert_eq!(value.is_structural_zero(),expected.marker,"{label} structural marker");
    for (index,(&actual,&expected)) in value.to_vec_f64().iter().zip(expected.values).enumerate() {
        assert!(actual==expected || actual.is_nan()&&expected.is_nan() || actual.is_finite()&&expected.is_finite()&&(actual-expected).abs()<=2e-6*(1.+expected.abs()),"{label}[{index}] {actual} != {expected}");
        if expected==0. {assert_eq!(actual.is_sign_negative(),expected.is_sign_negative(),"{label}[{index}] zero sign");}
    }
}

fn check_case(case: usize, golden: &Golden) {
        let mut x=tensor(golden.base,golden.shape,false);
        if golden.transpose {x=x.transpose(0,x.ndim()-1).unwrap();}
        let mut inputs=vec![x];
        if golden.kind=="prelu" {inputs.push(tensor(golden.weights,golden.weight_shape,false));}
        let mut directions=Vec::new();
        for (i,input) in inputs.iter().enumerate() {
            let mut shape=vec![2];shape.extend_from_slice(input.shape());
            directions.push(tensor(if i==0 {golden.seed_x}else{golden.seed_w},&shape,golden.wide));
        }
        let selected=inputs.iter().enumerate().filter_map(|(i,v)|(golden.mask&(1<<i)!=0).then_some(v)).collect::<Vec<_>>();
        let seeds=directions.iter().enumerate().filter_map(|(i,v)|(golden.mask&(1<<i)!=0).then_some(v)).collect::<Vec<_>>();
        let tangent=with_grad_enabled(golden.enabled,|| {
            let level=DualLevel::functional_batched(&selected,&seeds,2).unwrap();
            let output=if golden.kind=="prelu" {inputs[0].prelu(&inputs[1])}
                else {inputs[0].product((golden.dim!=-2).then_some(golden.dim as usize),golden.keep,Some(if golden.outwide {DType::F64}else{DType::F32}))}.unwrap();
            let tangent=level.batched_tangent(&output).unwrap().expect("selected floating operand has a genuine direction");
            assert_ne!(output._storage_ptr(),tangent._storage_ptr());
            tangent
        });
        let label=format!("case{case} {} mask{} wide{} enabled{}",golden.kind,golden.mask,golden.wide,golden.enabled);
        check(&tangent,&golden.output,&label);
        let queried=inputs.iter().chain(&directions).collect::<Vec<_>>();
        if tangent.requires_grad() {
            let seed=if tangent.dtype()==DType::F64 {Tensor::from_vec_f64(vec![1.],&[]).unwrap()}else{Tensor::scalar(1.)};
            let total=tangent.sum();
            let next=Tensor::vjp_many_wrt_optional(&[&total],&queried,&[&seed],false).unwrap();
            for (slot,(value,expected)) in next.iter().zip(golden.next).enumerate() {
                assert_eq!(value.is_some(),expected.is_some(),"{label} derivative{slot} presence");
                if let (Some(value),Some(expected))=(value,expected) {check(value,expected,&format!("{label} derivative{slot}"));}
            }
        } else {assert!(golden.next.iter().all(Option::is_none));}
        assert!(queried.iter().all(|v|v.grad().is_none()));
        assert!(!is_active());
}

#[test]
fn current_product_and_prelu_physical_rules_match_actual_torch() {
    let failures=GOLDENS.iter().enumerate().filter_map(|(case,golden)|std::panic::catch_unwind(||check_case(case,golden)).err().map(|_|case)).collect::<Vec<_>>();
    assert!(failures.is_empty(),"failed unchanged actualTorch configurations: {failures:?}");
    println!("{} actual Torch physical product/PReLU configurations",GOLDENS.len());
}
