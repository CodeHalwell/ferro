use ferro_core::{Tensor, Result};
use ferro_core::autograd::{with_grad_enabled,is_grad_enabled};
use ferro_core::forward_ad::{DualLevel,is_active};
struct Golden { shape:&'static[usize], axis:usize, ishape:&'static[usize], index:&'static[i64], kind:&'static str, enabled:bool,live:bool,data:&'static[f32],outshape:&'static[usize],stride:&'static[usize],value:&'static[f32],flag:bool,gradient:Option<&'static[f32]> }
include!("fixtures/physical_forward_index.rs");

fn evaluate(case:&Golden)->Result<(Tensor,Tensor)> {
 let x=Tensor::from_vec(case.data.to_vec(),case.shape)?.requires_grad_(case.live)?;
 let index=Tensor::from_vec_i64(case.index.to_vec(),case.ishape)?;
 let n=x.numel();let mut values=vec![0.;n*n];for i in 0..n {values[i*n+i]=1.;}
 let mut bshape=vec![n];bshape.extend_from_slice(case.shape);
 let basis=Tensor::from_vec(values,&[n,n])?.view(&bshape)?;
 let j=with_grad_enabled(case.enabled,||->Result<Tensor>{
  let level=DualLevel::functional_batched(&[&x],&[&basis],n)?;
  let y=x.alias_view()?.index_tensor(case.axis,&index)?;
  let y=match case.kind {"square"=>y.square()?,"transpose"=>y.transpose(0,1)?,_=>y};
  let tangent=level.batched_tangent(&y)?.unwrap();drop(level);
  let mut permutation=(1..tangent.ndim()).collect::<Vec<_>>();permutation.push(0);
  let mut shape=y.shape().to_vec();shape.extend_from_slice(x.shape());
  tangent.permute(&permutation)?.view(&shape)
 })?;
 Ok((x,j))
}

#[test]
fn array_index_aliases_real_leading_seed_layout_and_gradients_match_torch() {
 let mut failures=vec![];
 for (i,case) in GOLDENS.iter().enumerate() {
  let (x,j)=match evaluate(case) {Ok(value)=>value,Err(error)=>{failures.push(format!("{i}: {error:?}"));continue}};
  if j.shape()!=case.outshape||j.strides()!=case.stride||j.requires_grad()!=case.flag {failures.push(format!("{i}: {:?}/{:?}/{} != {:?}/{:?}/{}",j.shape(),j.strides(),j.requires_grad(),case.outshape,case.stride,case.flag));}
  for (a,b) in j.to_vec().iter().zip(case.value) {if (a-b).abs()>1e-5 {failures.push(format!("{i}: {a} != {b}"));break}}
  if let Some(expected)=case.gradient {
   let grad=j.sum().grad_wrt(&[&x],false).unwrap().remove(0);
   for (a,b) in grad.to_vec().iter().zip(expected) {if (a-b).abs()>1e-5 {failures.push(format!("{i}: gradient {a} != {b}"));break}}
  }
  assert!(x.grad().is_none());assert!(is_grad_enabled()&&!is_active());
 }
 assert!(failures.is_empty(),"{} failures:\n{}",failures.len(),failures.join("\n"));
 println!("{} actual Torch array Jacobian metadata/value/gradient cases passed",GOLDENS.len());
}
