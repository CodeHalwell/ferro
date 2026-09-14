pub use ferro_core::{Tensor, Error, Result, Device, DType};
use ferro_core::recurrent;
use recurrent::*;

#[test]
fn shared_weights_match_reference_and_truncation_cuts_only_history() {
    for trunc in [None,Some(2)] {
        let x=t(&[0.3,0.1,-0.2,0.4], &[4,1,1]).requires_grad_(true).unwrap();
        let wi=t(&[0.7], &[1,1]).requires_grad_(true).unwrap(); let wh=t(&[0.6], &[1,1]).requires_grad_(true).unwrap();
        let init=RecurrentState { h:t(&[0.2], &[1,1]).requires_grad_(true).unwrap(), c:None };
        let out=unroll(&x,&init,&[4],None,trunc,|x,s| Ok(RecurrentState { h:rnn_cell(x,&s.h,&wi,&wh,None,None)?, c:None })).unwrap();
        out.state.h.sum().backward();
        let gx=x.grad().unwrap().to_vec(); let gwi=wi.grad().unwrap().to_vec(); let gwh=wh.grad().unwrap().to_vec();
        if trunc.is_some() { assert_eq!(&gx[..2], &[0.0,0.0]); assert!(init.h.grad().is_none()); }
        else { assert!(gx[0].abs()>0.01); assert!(init.h.grad().unwrap().to_vec()[0].abs()>0.01); }
        x.zero_grad(); wi.zero_grad(); wh.zero_grad(); init.h.zero_grad();
        let mut h=init.h.clone();
        for i in 0..4 { if trunc==Some(2) && i==2 { h=h.detach_copy(); }
            let xx=x.index_select(0,&[i]).unwrap().reshape(&[1,1]).unwrap();
            h=xx.matmul(&wi.transpose(0,1).unwrap()).unwrap().add(&h.matmul(&wh.transpose(0,1).unwrap()).unwrap()).unwrap().tanh();
        }
        close(&out.state.h,&h.to_vec()); h.sum().backward();
        close(&wi.grad().unwrap(),&gwi); close(&wh.grad().unwrap(),&gwh); close(&x.grad().unwrap(),&gx);
    }
}
#[test]
fn delayed_copy_regression_trains_with_sgd_and_truncated_bptt() {
    use ferro_core::{params::Param, optim::Sgd};
    let wi=Param::new(t(&[0.3], &[1,1])); let wh=Param::new(t(&[0.5], &[1,1]));
    let mut opt=Sgd::new(vec![wi.clone(),wh.clone()],0.4);
    // Two signed impulses, then three blank timesteps; only final outputs supervised.
    // The detach at t=2 cuts input-encoding credit, while recurrent shared weights
    // still learn to retain the carried signal in the last two steps.
    let x=t(&[-0.4,0.4, 0.0,0.0, 0.0,0.0, 0.0,0.0], &[4,2,1]);
    let target=t(&[-0.4,0.4], &[2,1]);
    let init=RecurrentState { h:Tensor::zeros(&[2,1]), c:None };
    let mut first=0.0; let mut last=0.0;
    for epoch in 0..160 {
        opt.zero_grad();
        let a=wi.tensor(); let b=wh.tensor();
        let out=unroll(&x,&init,&[4,4],None,Some(2),|x,s| Ok(RecurrentState { h:rnn_cell(x,&s.h,&a,&b,None,None)?, c:None })).unwrap();
        let d=out.state.h.sub(&target).unwrap(); let loss=d.mul(&d).unwrap().mean();
        last=loss.to_vec()[0]; if epoch==0 { first=last; }
        loss.backward(); opt.step();
    }
    println!("delayed-copy TBPTT SGD: first={first:.8} last={last:.8}");
    assert!(last<first*0.02 && last<0.002, "{first} -> {last}");
}
#[test]
fn lstm_state_masking_and_explicit_detach() {
    let input=t(&[0.2,f32::NAN,0.3,f32::INFINITY], &[2,2,1]);
    let init=RecurrentState { h:t(&[0.1,0.8], &[2,1]), c:Some(t(&[0.2,0.9], &[2,1])) };
    let w=t(&[0.2,-0.3,0.4,0.1], &[4,1]);
    let out=unroll(&input,&init,&[2,0],None,None,|x,s| {
        let (h,c)=lstm_cell(x,&s.h,s.c.as_ref().unwrap(),&w,&w,None,None)?; Ok(RecurrentState { h,c:Some(c) })
    }).unwrap();
    assert_eq!(out.state.h.to_vec()[1],0.8); assert_eq!(out.state.c.as_ref().unwrap().to_vec()[1],0.9);
    let d=out.state.detach(); close(&d.h,&out.state.h.to_vec()); assert!(!d.h.requires_grad()); assert!(!d.c.unwrap().requires_grad());
    assert!(unroll(&input,&init,&[2,0],None,None,|_,_| Ok(RecurrentState { h:Tensor::zeros(&[1,1]), c:None })).is_err());
    assert!(rnn_cell(&Tensor::zeros(&[1]),&Tensor::zeros(&[1,1]),&w,&w,None,None).is_err());
    assert!(rnn_cell(&Tensor::zeros(&[1,1]),&Tensor::zeros(&[1,1]),&Tensor::ones(&[1,1]),&Tensor::ones(&[1,1]),Some(&Tensor::ones(&[1,1])),None).is_err());
    assert!(rnn_cell(&Tensor::from_vec_i64(vec![1],&[1,1]).unwrap(),&Tensor::zeros(&[1,1]),&w,&w,None,None).is_err());
}


#[test]
fn masked_unroll_skips_nan_padding_preserves_initial_and_resets() {
    let input=t(&[0.2,0.3,f32::NAN, 0.4,f32::NAN,f32::NAN, 0.5,f32::NAN,f32::NAN], &[3,3,1]).requires_grad_(true).unwrap();
    let initial=RecurrentState { h:t(&[0.1,0.2,0.7], &[3,1]), c:None };
    let wi=t(&[1.0], &[1,1]); let wh=t(&[0.5], &[1,1]);
    let step=|x:&Tensor,s:&RecurrentState| Ok(RecurrentState { h:rnn_cell(x,&s.h,&wi,&wh,None,None)?, c:None });
    let reset=[false,false,false, true,false,false, false,false,false];
    let out=unroll(&input,&initial,&[3,1,0],Some(&reset),None,step).unwrap();
    let first=0.25f32.tanh(); let second=0.45f32.tanh(); let last=(0.5+0.5*second).tanh();
    close(&out.outputs,&[first,0.4f32.tanh(),0.0,second,0.0,0.0,last,0.0,0.0]);
    close(&out.state.h,&[last,0.4f32.tanh(),0.7]);
    out.state.h.sum().backward(); let g=input.grad().unwrap().to_vec();
    for i in [0,2,4,5,7,8] { assert_eq!(g[i],0.0); }
    assert!(g[3]>0.0 && g[6]>0.0);
    assert!(unroll(&input,&initial,&[4,1,0],None,None,step).is_err());
    assert!(unroll(&input,&initial,&[3,1],None,None,step).is_err());
    assert!(unroll(&input,&initial,&[3,1,0],Some(&[false]),None,step).is_err());
    assert!(unroll(&input,&initial,&[3,1,0],None,Some(0),step).is_err());
    let empty=unroll(&Tensor::zeros(&[0,3,1]),&initial,&[0,0,0],None,None,step).unwrap();
    assert_eq!(empty.outputs.shape(), &[0,3,1]); close(&empty.state.h,&[0.1,0.2,0.7]);
}


#[test]
fn gated_cells_explicit_order_bias_and_finite_differences() {
    let x=t(&[0.3], &[1,1]); let h=t(&[-0.2], &[1,1]); let c=t(&[0.4], &[1,1]);
    let wi=t(&[0.2,-0.3,0.4,0.5], &[4,1]); let wh=t(&[-0.5,0.6,0.7,-0.2], &[4,1]);
    let bi=t(&[0.1,0.2,-0.1,0.3], &[4]); let bh=t(&[-0.2,0.1,0.3,-0.4], &[4]);
    let sigmoid=|v:f32| 1.0/(1.0+(-v).exp());
    let cc=sigmoid(0.09)*0.4+sigmoid(0.06)*0.18f32.tanh();
    let (hh,cs)=lstm_cell(&x,&h,&c,&wi,&wh,Some(&bi),Some(&bh)).unwrap();
    close(&cs,&[cc]); close(&hh,&[sigmoid(0.09)*cc.tanh()]);
    ferro_core::testkit::grad_check(&[x.clone(),h.clone(),c,wi,wh,bi,bh], |a| {
        let (h,c)=lstm_cell(&a[0],&a[1],&a[2],&a[3],&a[4],Some(&a[5]),Some(&a[6])).unwrap(); h.add(&c).unwrap()
    });
    let wi=t(&[0.2,-0.3,0.4], &[3,1]); let wh=t(&[-0.5,0.6,0.7], &[3,1]);
    let bi=t(&[0.1,0.2,-0.1], &[3]); let bh=t(&[-0.2,0.1,0.3], &[3]);
    let z=sigmoid(0.09); let n=(0.02+sigmoid(0.06)*0.16).tanh();
    close(&gru_cell(&x,&h,&wi,&wh,Some(&bi),Some(&bh)).unwrap(), &[(1.0-z)*n+z*(-0.2)]);
    ferro_core::testkit::grad_check(&[x,h,wi,wh,bi,bh], |a| gru_cell(&a[0],&a[1],&a[2],&a[3],Some(&a[4]),Some(&a[5])).unwrap());
}


fn t(v: &[f32], shape: &[usize]) -> Tensor { Tensor::from_vec(v.to_vec(), shape).unwrap() }
fn close(a: &Tensor, b: &[f32]) { assert_eq!(a.to_vec().len(), b.len()); for (x,y) in a.to_vec().iter().zip(b) { assert!((x-y).abs()<1e-5, "{x} != {y}"); } }
#[test]
fn rnn_explicit_affines_and_gradients() {
    let x=t(&[0.3], &[1,1]); let h=t(&[0.2], &[1,1]);
    let wi=t(&[0.7], &[1,1]); let wh=t(&[-0.4], &[1,1]);
    let bi=t(&[0.1], &[1]); let bh=t(&[-0.2], &[1]);
    close(&rnn_cell(&x,&h,&wi,&wh,Some(&bi),Some(&bh)).unwrap(), &[0.03f32.tanh()]);
    ferro_core::testkit::grad_check(&[x,h,wi,wh,bi,bh], |a| rnn_cell(&a[0],&a[1],&a[2],&a[3],Some(&a[4]),Some(&a[5])).unwrap());
}
