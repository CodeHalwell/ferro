use ferro_core::{DType,Tensor};
use ferro_core::forward_ad::DualLevel;
use ferro_core::autograd::with_grad_enabled;
fn leaf(values: &[f32]) -> Tensor {Tensor::from_vec(values.to_vec(),&[values.len()]).unwrap().requires_grad_(true).unwrap()}
fn equal(actual: &[f32], expected: &[f32]) {assert_eq!(actual.len(),expected.len());for (&a,&b) in actual.iter().zip(expected) {assert!(a.is_nan()&&b.is_nan()||a.to_bits()==b.to_bits(),"{a:?} != {b:?}");}}

#[test]
fn direct_prelu_jvp_matches_torch_zero_boundaries_and_missing_tangent_reverse_graphs() {
    // Fixed Torch 2.13 controls distinguish missing structural tangents from
    // ordinary allocated zeros, and ordinary (>0) from forward (>=0) branches.
    for mode in 0..3 {
        let x=leaf(&[f32::NEG_INFINITY,-2.,-0.,0.,2.,f32::INFINITY,f32::NAN]);
        let w=leaf(&[0.3]);let v=leaf(&[0.7;7]);let u=leaf(&[0.2]);
        let (primals,dirs)=match mode {0=>(vec![&x],vec![&v]),1=>(vec![&w],vec![&u]),_=>(vec![&x,&w],vec![&v,&u])};
        let level=DualLevel::new(&primals,&dirs).unwrap();
        let out=x.prelu(&w).unwrap();let tangent=level.tangent(&out).unwrap().unwrap();drop(level);
        let expected=match mode {0=>vec![0.21,0.21,0.7,0.7,0.7,0.7,0.21],1=>vec![f32::NEG_INFINITY,-0.4,0.,0.,0.,0.,f32::NAN],_=>vec![f32::NEG_INFINITY,-0.1899999976158142,0.7,0.7,0.7,0.7,f32::NAN]};
        // 0.3 * 0.7 rounds upward relative to the literal 0.21.
        let expected=if mode==0 {expected.iter().map(|&z|if z==0.21 {0.3f32*0.7}else{z}).collect()}else{expected};
        equal(&tangent.to_vec(),&expected);assert!(tangent.requires_grad());
        let seed=Tensor::ones(&[7]);
        let grads=Tensor::vjp_many_wrt_optional(&[&tangent],&[&x,&w,&v,&u],&[&seed],true).unwrap();
        equal(&grads[0].as_ref().unwrap().to_vec(),if mode==0 {&[0.;7]}else{&[0.2,0.2,0.,0.,0.,0.,0.2]});
        assert_eq!(grads[0].as_ref().unwrap().requires_grad(),mode!=0);
        equal(&grads[1].as_ref().unwrap().to_vec(),if mode==1 {&[0.]}else{&[2.0999999046325684]});
        assert_eq!(grads[1].as_ref().unwrap().requires_grad(),mode!=1);
        if mode!=1 {equal(&grads[2].as_ref().unwrap().to_vec(),&[0.3,0.3,1.,1.,1.,1.,0.3]);assert!(grads[2].as_ref().unwrap().requires_grad());}else{assert!(grads[2].is_none());}
        if mode!=0 {assert!(grads[3].as_ref().unwrap().to_vec()[0].is_nan());assert!(grads[3].as_ref().unwrap().requires_grad());}else{assert!(grads[3].is_none());}
        assert!(out.vjp_wrt(&[&x],&seed,true).is_err());
    }
}

#[test]
fn channel_jvp_mixed_directions_and_ambient_mode() {
    let x=leaf(&[-2.,0.,3.]);let v=leaf(&[0.3,0.7,-0.2]);
    let w=Tensor::from_vec_f64(vec![0.12345678912345678],&[1]).unwrap();
    let level=DualLevel::new(&[&x],&[&v]).unwrap();
    let output=x.to_dtype_autograd(DType::F64,false).unwrap().prelu(&w).unwrap();
    let tangent=level.tangent(&output).unwrap().unwrap();drop(level);
    assert_eq!(tangent.dtype(),DType::F64);
    assert_eq!(tangent.to_vec_f64(),vec![0.3f32 as f64*0.12345678912345678,0.7f32 as f64,-0.2f32 as f64]);
    for no_grad in [false,true] {for mode in 0..3 {
        let x=Tensor::from_vec(vec![-2.,0.,3.,-4.,-0.,5.],&[1,3,2]).unwrap().requires_grad_(true).unwrap();
        let w=leaf(&[0.2,0.3,0.4]);
        let v=Tensor::from_vec_f64(vec![0.7;6],&[1,3,2]).unwrap().requires_grad_(true).unwrap();
        let u=Tensor::from_vec_f64(vec![0.2;3],&[3]).unwrap().requires_grad_(true).unwrap();
        let (primals,dirs)=match mode {0=>(vec![&x],vec![&v]),1=>(vec![&w],vec![&u]),_=>(vec![&x,&w],vec![&v,&u])};
        let level=DualLevel::new(&primals,&dirs).unwrap();
        let out=with_grad_enabled(!no_grad,||x.prelu(&w).unwrap());
        let tangent=level.tangent(&out).unwrap().unwrap();drop(level);
        assert_eq!(tangent.dtype(),DType::F64);assert_eq!(tangent.requires_grad(),!no_grad);assert_eq!(out.requires_grad(),!no_grad);
        let expected=x.to_vec().into_iter().enumerate().map(|(i,z)|if z>=0. {if mode==1 {0.}else{0.7}}else{(if mode==1 {0.}else{w.to_vec()[i/2]as f64*0.7})+(if mode==0 {0.}else{0.2*z as f64})}).collect::<Vec<_>>();
        assert_eq!(tangent.to_vec_f64(),expected);
    }}
}

#[test]
fn empty_virtual_channel_jvp_does_not_materialize_slopes_and_error_scope_restores() {
    for shape in [[2,0,4],[2,3,0],[1,0,0]] {
        let x=Tensor::zeros(&shape);let w=Tensor::zeros(&[shape[1]]);
        let dx=Tensor::ones(&shape);let dw=Tensor::ones(&[shape[1]]);
        let level=DualLevel::new(&[&x,&w],&[&dx,&dw]).unwrap();
        let out=x.prelu(&w).unwrap();let tangent=level.tangent(&out).unwrap().unwrap();drop(level);
        assert_eq!(tangent.shape(),&shape);assert_eq!(tangent.strides(),x.strides());
    }
    let size=1usize<<40;
    let x=Tensor::from_vec(vec![],&[0,size,2]).unwrap();
    let w=Tensor::scalar(0.3).expand(&[size]).unwrap();
    let v=Tensor::from_vec(vec![],&[0,size,2]).unwrap();
    let level=DualLevel::new(&[&x],&[&v]).unwrap();
    let out=x.prelu(&w).unwrap();let tangent=level.tangent(&out).unwrap().unwrap();drop(level);
    assert_eq!(tangent.shape(),&[0,size,2]);assert!(!ferro_core::forward_ad::is_active());
    let scalar=Tensor::scalar(-2.);let scalar_dir=Tensor::scalar(0.7);
    let level=DualLevel::new(&[&scalar],&[&scalar_dir]).unwrap();
    let out=scalar.prelu(&Tensor::scalar(0.3)).unwrap();
    equal(&level.tangent(&out).unwrap().unwrap().to_vec(),&[0.3f32*0.7]);drop(level);
}

#[test]
fn direct_jvp_follows_predicate_physical_order_with_genuine_view_adjoints() {
    for transposed in [false,true] {for direction_transposed in [false,true] {for mode in 0..3 {for wide in [false,true] {
        let x=if transposed {Tensor::full(&[4,3,2],-2.).permute(&[2,1,0]).unwrap()}else{Tensor::full(&[2,3,4],-2.)};
        let x=x.requires_grad_(true).unwrap();let w=leaf(&[0.2,0.3,0.4]);
        let shape=if direction_transposed {[4,3,2]}else{[2,3,4]};
        let v=if wide {Tensor::from_vec_f64(vec![0.7;24],&shape).unwrap()}else{Tensor::full(&shape,0.7)};
        let v=if direction_transposed {v.permute(&[2,1,0]).unwrap()}else{v};let v=v.requires_grad_(true).unwrap();
        let u=if wide {Tensor::from_vec_f64(vec![0.2;3],&[3]).unwrap()}else{Tensor::full(&[3],0.2)}.requires_grad_(true).unwrap();
        let (primals,dirs)=match mode {0=>(vec![&x],vec![&v]),1=>(vec![&w],vec![&u]),_=>(vec![&x,&w],vec![&v,&u])};
        let level=DualLevel::new(&primals,&dirs).unwrap();let out=x.prelu(&w).unwrap();let tangent=level.tangent(&out).unwrap().unwrap();drop(level);
        assert_eq!(tangent.strides(),if transposed {&[1,2,6]}else{&[12,4,1]});
        if !wide {
            let seed=Tensor::ones(&[2,3,4]);
            let grads=Tensor::vjp_many_wrt_optional(&[&tangent],&[&x,&w,&v,&u],&[&seed],true).unwrap();
            equal(&grads[0].as_ref().unwrap().to_vec(),&vec![if mode==0 {0.}else{0.2};24]);
            if mode!=1 {
                let values=(0..24).map(|i|w.to_vec()[(i/4)%3]).collect::<Vec<_>>();equal(&grads[2].as_ref().unwrap().to_vec(),&values);
            }
        }
    }}}}
}
