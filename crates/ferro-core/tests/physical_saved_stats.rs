use ferro_core::{Tensor, Result};
use ferro_core::autograd::{with_grad_enabled, is_grad_enabled};
use ferro_core::forward_ad::{DualLevel, is_active};
struct Grad { values: &'static [f32], requires: bool, marker: bool }
struct Golden { physical: bool, mask: usize, enabled: bool, live: bool, seed_live: bool, requires: bool, values: &'static [f32], grads: [Option<Grad>; 6] }
include!("fixtures/physical2_layernorm_goldens.rs");

fn close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(),expected.len(),"{context}");
    for (index,(&a,&b)) in actual.iter().zip(expected).enumerate() {
        assert!((a-b).abs()<=5e-5+1e-4*b.abs(),"{context}[{index}]: {a} != {b}");
    }
}

fn evaluate(case: &Golden) -> Result<(Vec<Tensor>,Tensor)> {
    let shapes=[vec![2,3],vec![3],vec![3]];
    let mut values=Vec::new();
    for (i,shape) in shapes.iter().enumerate() {
        let n:usize=shape.iter().product();
        values.push(Tensor::from_vec((0..n).map(|v|v as f32*0.17+0.2+i as f32*0.1).collect(),shape)?.requires_grad_(case.live)?);
    }
    for shape in &shapes {
        let mut physical=shape.clone();if case.physical{physical.insert(0,3);}
        let n:usize=physical.iter().product();
        values.push(Tensor::from_vec((0..n).map(|v|v as f32*0.11-0.3).collect(),&physical)?.requires_grad_(case.seed_live)?);
    }
    let active=(0..3).filter(|i|case.mask>>i&1!=0).collect::<Vec<_>>();
    let output=with_grad_enabled(case.enabled,||->Result<Tensor>{
        let primals=active.iter().map(|&i|&values[i]).collect::<Vec<_>>();
        let seeds=active.iter().map(|&i|&values[i+3]).collect::<Vec<_>>();
        let level=if case.physical{DualLevel::functional_batched(&primals,&seeds,3)?}else{DualLevel::functional(&primals,&seeds)?};
        let output=values[0].layer_norm(Some(&values[1]),Some(&values[2]),1e-5)?;
        let mut tangents=level.tangents(&output)?.unwrap();
        assert_eq!(tangents.len(),1);
        Ok(tangents.remove(0))
    })?;
    Ok((values,output))
}

#[test]
fn original_statistics_match_normal_and_batched_torch_jvp_and_reverse_provenance() {
    for (index,case) in GOLDENS.iter().enumerate() {
        let (values,output)=evaluate(case).unwrap();
        assert_eq!(output.requires_grad(),case.requires,"case {index} flag");
        close(&output.to_vec(),case.values,&format!("case {index} tangent"));
        if output.requires_grad(){
            let inputs=values.iter().collect::<Vec<_>>();
            let gradients=output.sum().vjp_wrt_optional(&inputs,&Tensor::scalar(1.),true).unwrap();
            for (slot,(actual,expected)) in gradients.iter().zip(&case.grads).enumerate(){
                assert_eq!(actual.is_some(),expected.is_some(),"case {index} slot {slot} presence");
                if let(Some(actual),Some(expected))=(actual,expected){
                    close(&actual.to_vec(),expected.values,&format!("case {index} gradient {slot}"));
                    assert_eq!(actual.requires_grad(),expected.requires,"case {index} gradient {slot} flag");
                    assert_eq!(actual.is_structural_zero(),expected.marker,"case {index} gradient {slot} marker");
                }
            }
        }
        assert!(values.iter().all(|value|value.grad().is_none()));
        assert!(is_grad_enabled()&&!is_active());
    }
    println!("{} actual Torch saved-statistic JVP controls passed",GOLDENS.len());
}
