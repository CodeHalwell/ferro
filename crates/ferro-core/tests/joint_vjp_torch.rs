// Generated from actual Torch 2.13; regeneration script is retained in the evidence directory.
use ferro_core::{Tensor,DType};
fn close(actual:&Tensor, expected:&[f64], dtype:DType) {
 let got=if dtype==DType::F32 { actual.to_vec().into_iter().map(|v|v as f64).collect::<Vec<_>>() } else {actual.to_vec_f64()};
 assert_eq!(got.len(),expected.len());
 let tol=if dtype==DType::F32 {3e-6}else{3e-12};
 for(a,e)in got.iter().zip(expected) {assert!((a-e).abs() <= tol*(1.+e.abs()),"{got:?} != {expected:?}");}
}
#[test]
fn joint_torch_float32_0() {
 let dtype=DType::F32;
 let x=Tensor::from_vec(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&h,&z];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+0 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[-0.2177664041519165,-0.35870423913002014,-0.30980196595191956,-0.043555207550525665,-0.03782696649432182,-0.04556472972035408],dtype);
 close(got[1].as_ref().unwrap(),&[-0.19765764474868774,-0.18293586373329163],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float32_1() {
 let dtype=DType::F32;
 let x=Tensor::from_vec(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&h,&z];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+1 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),true).unwrap();
 close(got[0].as_ref().unwrap(),&[-0.24498720467090607,-0.40354228019714355,-0.348527193069458,-0.0489996001124382,-0.04255533218383789,-0.05126031115651131],dtype);
 close(got[1].as_ref().unwrap(),&[-0.22236484289169312,-0.20580285787582397],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 let total=got[0].as_ref().unwrap().sum().add(&got[1].as_ref().unwrap().sum()).unwrap();
let mut selected=vec![&x,&y];selected.extend(seeds.iter());
let next=total.vjp_wrt_optional(&selected,&Tensor::scalar(1.),false).unwrap();
close(next[0].as_ref().unwrap(),&[-0.1889781653881073,-0.4425090253353119,0.08262351155281067,-0.30658861994743347,-0.3678703308105469,-0.2828572690486908],dtype);
close(next[1].as_ref().unwrap(),&[-0.39880964159965515,-1.0924943685531616],dtype);
close(next[2].as_ref().unwrap(),&[0.07419590651988983,0.9748306274414062,1.0530190467834473,0.19872544705867767,-0.4982043206691742,0.3942948877811432],dtype);
close(next[3].as_ref().unwrap(),&[7.347545146942139],dtype);
}
#[test]
fn joint_torch_float32_2() {
 let dtype=DType::F32;
 let x=Tensor::from_vec(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&h,&z];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+2 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 outputs.reverse();seeds.reverse();
let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[-0.27220797538757324,-0.4483802914619446,-0.38725244998931885,-0.05444400757551193,-0.04728370159864426,-0.05695590376853943],dtype);
 close(got[1].as_ref().unwrap(),&[-0.24707207083702087,-0.22866983711719513],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float32_3() {
 let dtype=DType::F32;
 let x=Tensor::from_vec(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&h,&z];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+3 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 outputs.reverse();seeds.reverse();
let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),true).unwrap();
 close(got[0].as_ref().unwrap(),&[-0.2994288206100464,-0.4932183027267456,-0.4259777069091797,-0.05988841503858566,-0.052012085914611816,-0.06265150755643845],dtype);
 close(got[1].as_ref().unwrap(),&[-0.27177929878234863,-0.2515367865562439],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 let total=got[0].as_ref().unwrap().sum().add(&got[1].as_ref().unwrap().sum()).unwrap();
let mut selected=vec![&x,&y];selected.extend(seeds.iter());
let next=total.vjp_wrt_optional(&selected,&Tensor::scalar(1.),false).unwrap();
close(next[0].as_ref().unwrap(),&[-0.23097330331802368,-0.5408443808555603,0.10098427534103394,-0.37471941113471985,-0.44961926341056824,-0.34571442008018494],dtype);
close(next[1].as_ref().unwrap(),&[-0.4874340891838074,-1.3352707624435425],dtype);
close(next[2].as_ref().unwrap(),&[7.347545146942139],dtype);
close(next[3].as_ref().unwrap(),&[0.07419590651988983,0.9748306274414062,1.0530190467834473,0.19872544705867767,-0.4982043206691742,0.3942948877811432],dtype);
}
#[test]
fn joint_torch_float32_4() {
 let dtype=DType::F32;
 let x=Tensor::from_vec(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&z,&h,&z,&q,&x];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+4 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[0.8089813590049744,0.609724760055542,0.41524019837379456,0.38086384534835815,0.3973810374736786,0.37672892212867737],dtype);
 close(got[1].as_ref().unwrap(),&[-0.3132822513580322,0.2791571617126465],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float32_5() {
 let dtype=DType::F32;
 let x=Tensor::from_vec(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&z,&h,&z,&q,&x];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+5 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),true).unwrap();
 close(got[0].as_ref().unwrap(),&[0.8763964772224426,0.6605351567268372,0.44984349608421326,0.4126024842262268,0.43049609661102295,0.4081229567527771],dtype);
 close(got[1].as_ref().unwrap(),&[-0.3393891751766205,0.3024202585220337],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 let total=got[0].as_ref().unwrap().sum().add(&got[1].as_ref().unwrap().sum()).unwrap();
let mut selected=vec![&x,&y];selected.extend(seeds.iter());
let next=total.vjp_wrt_optional(&selected,&Tensor::scalar(1.),false).unwrap();
close(next[0].as_ref().unwrap(),&[0.49700456857681274,-0.3327628970146179,-0.3072638511657715,0.397765576839447,0.21706938743591309,0.4398285746574402],dtype);
close(next[1].as_ref().unwrap(),&[0.21583890914916992,0.7790806293487549],dtype);
close(next[2].as_ref().unwrap(),&[7.347545146942139],dtype);
close(next[3].as_ref().unwrap(),&[0.07419590651988983,0.9748306274414062,1.0530190467834473,0.19872544705867767,-0.4982043206691742,0.3942948877811432],dtype);
close(next[4].as_ref().unwrap(),&[7.347545146942139],dtype);
close(next[5].as_ref().unwrap(),&[2.196861505508423],dtype);
close(next[6].as_ref().unwrap(),&[1.0,1.0,1.0,1.0,1.0,1.0],dtype);
}
#[test]
fn joint_torch_float32_6() {
 let dtype=DType::F32;
 let x=Tensor::from_vec(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&z,&h,&z,&q,&x];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+6 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 outputs.reverse();seeds.reverse();
let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[0.9438115954399109,0.7113455533981323,0.4844469130039215,0.44434118270874023,0.4636112451553345,0.439517080783844],dtype);
 close(got[1].as_ref().unwrap(),&[-0.3654960095882416,0.3256833553314209],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float32_7() {
 let dtype=DType::F32;
 let x=Tensor::from_vec(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&z,&h,&z,&q,&x];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+7 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 outputs.reverse();seeds.reverse();
let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),true).unwrap();
 close(got[0].as_ref().unwrap(),&[1.011226773262024,0.7621559500694275,0.5190502405166626,0.4760797917842865,0.49672627449035645,0.4709111154079437],dtype);
 close(got[1].as_ref().unwrap(),&[-0.39160293340682983,0.3489464521408081],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 let total=got[0].as_ref().unwrap().sum().add(&got[1].as_ref().unwrap().sum()).unwrap();
let mut selected=vec![&x,&y];selected.extend(seeds.iter());
let next=total.vjp_wrt_optional(&selected,&Tensor::scalar(1.),false).unwrap();
close(next[0].as_ref().unwrap(),&[0.5734668970108032,-0.383957177400589,-0.35453516244888306,0.45896026492118835,0.25046464800834656,0.5074944496154785],dtype);
close(next[1].as_ref().unwrap(),&[0.24904489517211914,0.8989390730857849],dtype);
close(next[2].as_ref().unwrap(),&[1.0,1.0,1.0,1.0,1.0,1.0],dtype);
close(next[3].as_ref().unwrap(),&[2.196861505508423],dtype);
close(next[4].as_ref().unwrap(),&[7.347545146942139],dtype);
close(next[5].as_ref().unwrap(),&[0.07419590651988983,0.9748306274414062,1.0530190467834473,0.19872544705867767,-0.4982043206691742,0.3942948877811432],dtype);
close(next[6].as_ref().unwrap(),&[7.347545146942139],dtype);
}
#[test]
fn joint_torch_float64_0() {
 let dtype=DType::F64;
 let x=Tensor::from_vec_f64(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec_f64(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&h,&z];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+0 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[-0.21776640001728148,-0.35870422448364125,-0.30980193998051897,-0.04355520874595128,-0.037826967492317495,-0.045564729632122464],dtype);
 close(got[1].as_ref().unwrap(),&[-0.19765763858387286,-0.1829358448502062],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float64_1() {
 let dtype=DType::F64;
 let x=Tensor::from_vec_f64(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec_f64(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&h,&z];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+1 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[-0.24498720001944163,-0.4035422525440964,-0.34852718247808384,-0.048999609839195185,-0.04255533842885717,-0.05126032083613777],dtype);
 close(got[1].as_ref().unwrap(),&[-0.22236484340685697,-0.20580282545648199],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float64_2() {
 let dtype=DType::F64;
 let x=Tensor::from_vec_f64(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec_f64(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&h,&z];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+2 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 outputs.reverse();seeds.reverse();
let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[-0.27220800002160184,-0.4483802806045516,-0.3872524249756487,-0.054444010932439085,-0.047283709365396845,-0.05695591204015309],dtype);
 close(got[1].as_ref().unwrap(),&[-0.24707204822984108,-0.22866980606275772],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float64_3() {
 let dtype=DType::F64;
 let x=Tensor::from_vec_f64(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec_f64(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&h,&z];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+3 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 outputs.reverse();seeds.reverse();
let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[-0.29942880002376204,-0.4932183086650067,-0.42597766747321353,-0.05988841202568297,-0.052012080301936534,-0.06265150324416838],dtype);
 close(got[1].as_ref().unwrap(),&[-0.27177925305282513,-0.25153678666903356],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float64_4() {
 let dtype=DType::F64;
 let x=Tensor::from_vec_f64(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec_f64(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&z,&h,&z,&q,&x];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+4 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[0.8089813705079606,0.6097247525498761,0.4152401756012276,0.3808638438508669,0.39738102814527143,0.3767289022160857],dtype);
 close(got[1].as_ref().unwrap(),&[-0.3132823135056155,0.2791571478679165],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float64_5() {
 let dtype=DType::F64;
 let x=Tensor::from_vec_f64(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec_f64(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&z,&h,&z,&q,&x];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+5 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[0.8763964847169574,0.660535148595699,0.4498435235679965,0.4126024975051058,0.430496113824044,0.4081229774007594],dtype);
 close(got[1].as_ref().unwrap(),&[-0.3393891729644169,0.30242024352357616],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float64_6() {
 let dtype=DType::F64;
 let x=Tensor::from_vec_f64(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec_f64(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&z,&h,&z,&q,&x];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+6 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 outputs.reverse();seeds.reverse();
let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[0.9438115989259539,0.711345544641522,0.4844468715347654,0.4443411511593447,0.46361119950281665,0.4395170525854332],dtype);
 close(got[1].as_ref().unwrap(),&[-0.3654960324232181,0.3256833391792359],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
#[test]
fn joint_torch_float64_7() {
 let dtype=DType::F64;
 let x=Tensor::from_vec_f64(vec![-0.7,0.2,0.9,0.4,-0.3,0.6],&[2,3]).unwrap().requires_grad_(true).unwrap();
 let y=Tensor::from_vec_f64(vec![0.8,-0.2],&[2,1]).unwrap().requires_grad_(true).unwrap();
 let h=x.mul(&y).unwrap().tanh();let z=h.mul(&h).unwrap().add(&x).unwrap().sum();let q=h.sum();
 let mut outputs=vec![&z,&h,&z,&q,&x];
 let mut seeds=outputs.iter().enumerate().map(|(i,v)|{
  let data=vec![(-0.3+0.2*i as f64)*(1.+7 as f64*0.125);v.numel()];
  let seed=if dtype==DType::F32{Tensor::from_vec(data.iter().map(|v|*v as f32).collect(),v.shape()).unwrap()}else{Tensor::from_vec_f64(data,v.shape()).unwrap()};
  seed.requires_grad_(true).unwrap()
 }).collect::<Vec<_>>();
 outputs.reverse();seeds.reverse();
let got=Tensor::vjp_many_wrt_optional(&outputs,&[&x,&y],&seeds.iter().collect::<Vec<_>>(),false).unwrap();
 close(got[0].as_ref().unwrap(),&[1.0112267131349506,0.762155940687345,0.5190502195015344,0.4760798048135836,0.49672628518158923,0.470911127770107],dtype);
 close(got[1].as_ref().unwrap(),&[-0.39160289188201935,0.3489464348348957],dtype);
 assert!(x.grad().is_none()&&y.grad().is_none());
 }
