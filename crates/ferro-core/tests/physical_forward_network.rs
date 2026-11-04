use ferro_core::{Tensor, Result, Device};
use ferro_core::autograd::{with_grad_enabled, is_grad_enabled};
use ferro_core::forward_ad::{DualLevel, is_active};

struct Block { shape: &'static [usize], strides: &'static [usize], data: &'static [f32], flag: bool, gradient: [Option<&'static [f32]>;2] }
struct NetworkGolden { enabled: bool, xg: bool, wg: bool, selected: bool, blocks: &'static [Block], aliases: &'static [bool] }
include!("fixtures/physical_forward_network.rs");

fn evaluate(golden: &NetworkGolden) -> Result<([Tensor;2], Vec<Tensor>)> {
    let x=Tensor::from_vec(vec![0.2,-0.4,0.3,0.5],&[2,2])?.requires_grad_(golden.xg)?;
    let w=Tensor::from_vec(vec![0.6,-0.3,0.4,0.1],&[2,2])?.requires_grad_(golden.wg)?;
    let n=if golden.selected {8} else {4};
    let mut entries=vec![0.;n*n];for i in 0..n {entries[i*n+i]=1.;}
    let basis=Tensor::from_vec(entries,&[n,n])?;
    let first=basis.narrow(1,0,4)?.view(&[n,2,2])?;
    let second=if golden.selected {Some(basis.narrow(1,4,4)?.view(&[n,2,2])?)} else {None};
    let primals=if golden.selected {vec![&x,&w]} else {vec![&x]};
    let directions=if let Some(second)=&second {vec![&first,second]} else {vec![&first]};
    let results=with_grad_enabled(golden.enabled,|| -> Result<Vec<Tensor>> {
        let level=DualLevel::functional_batched(&primals,&directions,n)?;
        let h=x.matmul(&w)?.tanh();
        let outputs=vec![h.clone(),h.transpose(0,1)?,h.softmax(1)?,h.sum(),h.clone()];
        let tangents=outputs.iter().map(|output|level.tangents(output)).collect::<Result<Vec<_>>>()?;
        drop(level);
        let mut results=vec![];
        for (output,tangent) in outputs.iter().zip(tangents) {
            let mut physical=vec![n];physical.extend_from_slice(output.shape());
            let value=if let Some(mut values)=tangent {values.remove(0)} else {Tensor::full_on(output.shape(),0.,Device::Cpu)?.expand(&physical)?};
            let mut axes=(1..=output.ndim()).collect::<Vec<_>>();axes.push(0);
            let value=value.permute(&axes)?;
            for (i,input) in primals.iter().enumerate() {
                let mut shape=output.shape().to_vec();shape.extend_from_slice(input.shape());
                results.push(value.narrow(value.ndim()-1,i*4,4)?.view(&shape)?);
            }
        }
        Ok(results)
    })?;
    Ok(([x,w],results))
}

#[test]
fn actual_torch_shared_network_blocks_storage_and_captured_gradients() {
    let mut failures=vec![];
    for (case,golden) in NETWORKS.iter().enumerate() {
        let (inputs,blocks)=evaluate(golden).unwrap();
        for (index,(block,expected)) in blocks.iter().zip(golden.blocks).enumerate() {
            if block.shape()!=expected.shape || block.strides()!=expected.strides || block.requires_grad()!=expected.flag {
                failures.push(format!("case{case} block{index} metadata {:?}/{:?}/{} vs {:?}/{:?}/{}",block.shape(),block.strides(),block.requires_grad(),expected.shape,expected.strides,expected.flag));
            }
            for (a,b) in block.to_vec().iter().zip(expected.data) { if (a-b).abs()>2e-5 { failures.push(format!("case{case} block{index} value {a}!={b}"));break; } }
            if block.requires_grad() {
                let derivatives=block.sum().vjp_wrt_optional(&[&inputs[0],&inputs[1]],&Tensor::scalar(1.),false).unwrap();
                for (input,(actual,expected)) in derivatives.iter().zip(expected.gradient).enumerate() {
                    match (actual,expected) {
                        (None,None)=>{},
                        (Some(actual),Some(expected))=>for (a,b) in actual.to_vec().iter().zip(expected) {if (a-b).abs()>2e-5 {failures.push(format!("case{case} block{index} gradient{input} {a}!={b}"));break;}},
                        _=>failures.push(format!("case{case} block{index} gradient{input} presence differs")),
                    }
                }
            }
        }
        for (left,a) in blocks.iter().enumerate() {for (right,b) in blocks.iter().enumerate() {
            if (a._storage_ptr()==b._storage_ptr())!=golden.aliases[left*blocks.len()+right] {failures.push(format!("case{case} storage {left}/{right} differs"));}
        }}
        for input in &inputs {assert!(input.grad().is_none());}
        assert!(is_grad_enabled()&&!is_active());
    }
    assert!(failures.is_empty(),"{} failures:\n{}",failures.len(),failures.join("\n"));
    println!("{} actual Torch shared network cases and {} Jacobian blocks passed",NETWORKS.len(),NETWORKS.iter().map(|x|x.blocks.len()).sum::<usize>());
}
