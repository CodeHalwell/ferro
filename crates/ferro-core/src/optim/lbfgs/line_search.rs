//! Bracketing and zoom for the strong Wolfe conditions, CPU f32 scalars.
use super::{dot,invalid,max_abs};
use crate::Result;
#[derive(Clone)]
pub(super) struct Point {pub t:f32,pub loss:f32,pub grad:Vec<f32>,slope:f32}
fn cubic(a:&Point,b:&Point,bounds:Option<(f32,f32)>)->f32 {
 let (lo,hi)=bounds.unwrap_or((a.t.min(b.t),a.t.max(b.t)));
 let d1=a.slope+b.slope-3.*(a.loss-b.loss)/(a.t-b.t);
 let square=d1*d1-a.slope*b.slope;
 if square>=0. {
  let d2=square.sqrt();
  let proposal=if a.t<=b.t {b.t-(b.t-a.t)*(b.slope+d2-d1)/(b.slope-a.slope+2.*d2)}
   else {a.t-(a.t-b.t)*(a.slope+d2-d1)/(a.slope-b.slope+2.*d2)};
  if proposal.is_nan(){proposal}else{proposal.max(lo).min(hi)}
 }else{(lo+hi)/2.}
}
pub(super) fn search<F:FnMut(f32)->Result<(f32,Vec<f32>)>>(initial:f32,d:&[f32],loss:f32,gradient:&[f32],slope:f32,budget:u32,mut evaluate:F)->Result<Point>{
 let eval=|t:f32,evaluate:&mut F|->Result<Point>{
  if !t.is_finite() || t<0. {return Err(invalid("line search proposed a nonfinite or negative step"));}
  let (loss,grad)=evaluate(t)?;let slope=dot(&grad,d);Ok(Point{t,loss,grad,slope})
 };
 let base=Point{t:0.,loss,grad:gradient.to_vec(),slope};
 let mut current=eval(initial,&mut evaluate)?;let mut previous=base.clone();let mut iteration=0;
 let mut bracket=None;
 while iteration<budget {
  if current.loss>loss+1e-4*current.t*slope || (iteration>1 && current.loss>=previous.loss) {
   bracket=Some([previous.clone(),current.clone()]);break;
  }
  if current.slope.abs()<=-0.9*slope{return Ok(current);}
  if current.slope>=0. {bracket=Some([previous.clone(),current.clone()]);break;}
  let proposal=cubic(&previous,&current,Some((current.t+0.01*(current.t-previous.t),current.t*10.)));
  previous=current;current=eval(proposal,&mut evaluate)?;iteration+=1;
 }
 let Some(mut bracket)=bracket else {
  return Ok(if base.loss<=current.loss {base}else{current});
 };
 let mut insufficient=false;let mut low=if bracket[0].loss<=bracket[1].loss {0}else{1};let mut high=1-low;
 while iteration<budget {
  if (bracket[1].t-bracket[0].t).abs()*max_abs(d)<1e-9 {break;}
  let mut proposal=cubic(&bracket[0],&bracket[1],None);
  let lower=bracket[0].t.min(bracket[1].t);let upper=bracket[0].t.max(bracket[1].t);let eps=0.1*(upper-lower);
  if (upper-proposal).min(proposal-lower)<eps {
   if insufficient || proposal>=upper || proposal<=lower {
    proposal=if (proposal-upper).abs()<(proposal-lower).abs(){upper-eps}else{lower+eps};insufficient=false;
   }else{insufficient=true;}
  }else{insufficient=false;}
  let point=eval(proposal,&mut evaluate)?;iteration+=1;
  if point.loss>loss+1e-4*point.t*slope || point.loss>=bracket[low].loss {
   bracket[high]=point;low=if bracket[0].loss<=bracket[1].loss {0}else{1};high=1-low;
  }else{
   let done=point.slope.abs()<=-0.9*slope;
   if !done && point.slope*(bracket[high].t-bracket[low].t)>=0. {bracket[high]=bracket[low].clone();}
   bracket[low]=point;if done{break;}
  }
 }
 Ok(bracket[low].clone())
}
