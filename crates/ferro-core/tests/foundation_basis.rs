pub use ferro_core::{Tensor, Device, DType, Error, Result, Param};
use ferro_core::basis;
use basis::{BSplineBasis, OutsideDomain, kan};

// Bernstein polynomials are an independent closed-form oracle for clamped
// single-span cubic B-splines (no Cox-de Boor recurrence in this oracle).
fn bernstein(x: f64) -> ([f64;4], [f64;4]) {
    let u = 1.-x;
    ([u*u*u, 3.*x*u*u, 3.*x*x*u, x*x*x],
     [-3.*u*u, 3.*u*u-6.*x*u, 6.*x*u-3.*x*x, 3.*x*x])
}
fn cubic() -> BSplineBasis {
    BSplineBasis::new(vec![0.,0.,0.,0.,1.,1.,1.,1.], 3, OutsideDomain::Zero).unwrap()
}

#[test]
fn kan_forward_input_and_coefficient_adjoint_oracle() {
    let b = cubic();
    let xv = vec![0.13,0.29,0.47,0.61,0.73,0.89];
    let cv: Vec<f32> = (0..24).map(|i| ((i*7%19) as f32-9.)/7.).collect();
    let x = Tensor::from_vec(xv.clone(), &[3,2]).unwrap().requires_grad_(true).unwrap();
    let c = Tensor::from_vec(cv.clone(), &[3,2,4]).unwrap().requires_grad_(true).unwrap();
    let seedv = vec![0.3,-0.7,1.1,-0.2,0.8,0.4,0.6,-1.2,0.9];
    let seed = Tensor::from_vec(seedv.clone(), &[3,3]).unwrap();
    let y = kan(&x,&c,&b).unwrap();
    let gs = y.vjp_wrt(&[&x,&c], &seed, false).unwrap();
    let mut ey = vec![0f64;9]; let mut ex = vec![0f64;6]; let mut ec = vec![0f64;24];
    for r in 0..3 { for j in 0..3 { for i in 0..2 {
        let (bv,db) = bernstein(xv[r*2+i] as f64);
        for k in 0..4 {
            let ci = (j*2+i)*4+k;
            ey[r*3+j] += cv[ci] as f64*bv[k];
            ex[r*2+i] += seedv[r*3+j] as f64*cv[ci] as f64*db[k];
            ec[ci] += seedv[r*3+j] as f64*bv[k];
        }
    }}}
    for (actual,expected) in [(y.to_vec(),ey),(gs[0].to_vec(),ex),(gs[1].to_vec(),ec)] {
        for (a,e) in actual.iter().zip(expected) { assert!((*a as f64-e).abs()<2e-6, "{a} != {e}"); }
    }
    ferro_core::testkit::grad_check(&[x,c], |ts| kan(&ts[0],&ts[1],&b).unwrap().mul(&seed).unwrap().sum());
}



#[test]
fn two_layer_nonlinear_regression_shared_adam() {
    use basis::KanLayer;
    let b = cubic();
    let first = KanLayer::from_coefficients(b.clone(), Tensor::from_vec(vec![0.15,0.3,0.5,0.7, 0.7,0.5,0.3,0.15], &[2,1,4]).unwrap()).unwrap();
    let outer_basis = BSplineBasis::new(vec![-3.,-3.,-3.,-3.,3.,3.,3.,3.],3,OutsideDomain::Error).unwrap();
    let second = KanLayer::from_coefficients(outer_basis, Tensor::from_vec(vec![0.1,0.3,-0.2,0.2, -0.1,0.2,0.4,-0.2], &[1,2,4]).unwrap()).unwrap();
    assert_eq!(first.basis().degree(),3);
    let p1 = first.coefficients(); let p2 = second.coefficients();
    let before1 = p1.tensor().to_vec(); let before2 = p2.tensor().to_vec();
    let mut opt = ferro_core::optim::Adam::new(vec![p1.clone(),p2.clone()],0.01);
    let xv: Vec<f32> = (0..41).map(|i| 0.05+0.9*i as f32/40.).collect();
    let target = |x:f32| 0.3+0.5*(4.*x).sin();
    let x = Tensor::from_vec(xv.clone(), &[41,1]).unwrap();
    let y = Tensor::from_vec(xv.iter().map(|&v| target(v)).collect(), &[41,1]).unwrap();
    let loss = || { let d=second.forward(&first.forward(&x).unwrap()).unwrap().sub(&y).unwrap(); d.mul(&d).unwrap().mean() };
    let initial = loss().to_vec()[0];
    for step in 0..600 {
        opt.zero_grad(); let l=loss(); l.backward();
        if step==0 {
            for p in [&p1,&p2] { assert!(p.grad().unwrap().to_vec().iter().any(|v| v.abs()>1e-6)); }
        }
        opt.step();
    }
    let final_loss=loss().to_vec()[0];
    let held:Vec<f32>=(0..40).map(|i| 0.05+0.9*(i as f32+0.5)/40.).collect();
    let pred=second.forward(&first.forward(&Tensor::from_vec(held.clone(), &[40,1]).unwrap()).unwrap()).unwrap().to_vec();
    let held_mse=pred.iter().zip(&held).map(|(&p,&v)| (p-target(v)).powi(2)).sum::<f32>()/40.;
    println!("training initial={initial:.9} final={final_loss:.9} held_out={held_mse:.9}");
    assert!(final_loss < initial*0.01 && final_loss<1e-4);
    assert!(held_mse<1e-4);
    assert_ne!(before1,p1.tensor().to_vec()); assert_ne!(before2,p2.tensor().to_vec());
}


#[test]
fn checked_contracts_degree_zero_nonclamped_and_empty() {
    for (t,p) in [(vec![],0),(vec![0.,1.],usize::MAX),(vec![0.,1.,0.,1.],1),(vec![0.,0.,0.,0.],1),(vec![0.,f32::NAN],0),(vec![0.,f32::INFINITY],0)] {
        assert!(BSplineBasis::new(t,p,OutsideDomain::Zero).is_err());
    }
    let b=BSplineBasis::new(vec![0.,0.5,1.],0,OutsideDomain::Error).unwrap();
    assert_eq!(b.knots(), &[0.,0.5,1.]);
    assert_eq!(b.domain(),(0.,1.));
    let x=Tensor::from_vec(vec![0.,0.25,0.5,1.], &[2,2]).unwrap().requires_grad_(true).unwrap();
    let y=b.evaluate(&x).unwrap();
    assert_eq!(y.to_vec(),vec![1.,0.,1.,0.,0.,1.,0.,1.]);
    y.sum().backward(); assert_eq!(x.grad().unwrap().to_vec(),vec![0.;4]);
    for v in [-0.1,1.1,f32::NAN,f32::INFINITY] { assert!(b.evaluate(&Tensor::scalar(v)).is_err()); }
    assert!(matches!(b.evaluate(&Tensor::from_vec_f64(vec![0.5], &[1]).unwrap()),Err(Error::DtypeMismatch{..})));
    let nonclamped=BSplineBasis::new(vec![-1.,0.,1.,2.,3.],1,OutsideDomain::Zero).unwrap();
    assert_eq!(nonclamped.evaluate(&Tensor::from_vec(vec![0.,2.], &[2]).unwrap()).unwrap().to_vec(),vec![1.,0.,0.,0.,0.,1.]);
    assert_eq!(cubic().evaluate(&Tensor::zeros(&[0,2])).unwrap().shape(), &[0,2,4]);
    assert_eq!(cubic().evaluate(&Tensor::scalar(0.5)).unwrap().shape(), &[4]);
    let c=Tensor::ones(&[2,1,4]);
    for shape in [vec![],vec![1],vec![1,2],vec![1,1,1]] { assert!(kan(&Tensor::zeros(&shape),&c,&cubic()).is_err()); }
    assert!(basis::KanLayer::from_coefficients(cubic(),Tensor::zeros(&[2,1,3])).is_err());
    let leaf=Tensor::ones(&[2,1,4]).requires_grad_(true).unwrap();
    assert!(basis::KanLayer::from_coefficients(cubic(),leaf.mul(&leaf).unwrap()).is_err());
}

#[test]
fn basis_vjp_multispan_finite_differences_and_boundary_convention() {
    for degree in 0..=4 {
        let mut knots=vec![0.;degree+1]; knots.extend([0.3,0.6,0.6]); knots.extend(vec![1.;degree+1]);
        let b=BSplineBasis::new(knots,degree,OutsideDomain::Zero).unwrap();
        let x=Tensor::from_vec(vec![-0.1,0.13,0.43,0.79,1.1], &[5]).unwrap();
        let weights=Tensor::from_vec((0..5*b.num_basis()).map(|i| (i as f32*0.7).sin()).collect(), &[5,b.num_basis()]).unwrap();
        ferro_core::testkit::grad_check(&[x], |ts| b.evaluate(&ts[0]).unwrap().mul(&weights).unwrap().sum());
    }
    let b=BSplineBasis::new(vec![0.,0.,0.5,1.,1.],1,OutsideDomain::Zero).unwrap();
    let x=Tensor::from_vec(vec![-0.1,0.,0.5,1.,1.1], &[5]).unwrap().requires_grad_(true).unwrap();
    let w=Tensor::from_vec(vec![1.,2.,4.], &[3]).unwrap();
    b.evaluate(&x).unwrap().mul(&w).unwrap().sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(),vec![0.,2.,4.,4.,0.]);
}

#[test]
fn higher_order_input_rejected_coefficient_graph_preserved() {
    let x=Tensor::from_vec(vec![0.25], &[1,1]).unwrap().requires_grad_(true).unwrap();
    let c=Tensor::from_vec(vec![0.1,0.4,0.2,0.8], &[1,1,4]).unwrap().requires_grad_(true).unwrap();
    let y=kan(&x,&c,&cubic()).unwrap();
    assert!(matches!(y.sum().grad_wrt(&[&x],true),Err(Error::Unsupported{..})));
    let loss=y.mul(&y).unwrap().sum();
    let dc=loss.grad_wrt(&[&c],true).unwrap().remove(0);
    assert!(dc.requires_grad());
    let second=dc.sum().grad_wrt(&[&c],false).unwrap().remove(0).to_vec();
    let (bv,_)=bernstein(0.25);
    for (a,b) in second.iter().zip(bv) { assert!((*a as f64-2.*b).abs()<1e-6); }
}

#[test]
fn cubic_basis_distribution_ulp_oracle() {
    let xv:Vec<f32>=(0..1001).map(|i| i as f32/1000.).collect();
    let y=cubic().evaluate(&Tensor::from_vec(xv.clone(), &[1001]).unwrap()).unwrap().to_vec();
    let mut ulps=Vec::new();
    for (&x,row) in xv.iter().zip(y.chunks(4)) {
        for (&a,e) in row.iter().zip(bernstein(x as f64).0) {
            let expected=e as f32;
            assert!(a.is_finite() && a>=0.);
            // Both operands are nonnegative; signed zeros are collapsed.
            let ab=if a==0. {0} else {a.to_bits()};
            let eb=if expected==0. {0} else {expected.to_bits()};
            ulps.push(ab.abs_diff(eb));
        }
    }
    ulps.sort_unstable();
    println!("bernstein_basis n={} ulp_p50={} ulp_p95={} ulp_p99={} ulp_max={}",ulps.len(),ulps[2002],ulps[3803],ulps[3963],ulps[4003]);
    assert!(*ulps.last().unwrap()<=1);
}

#[test]
fn devices_rejected_before_download() {
    use ferro_core::dispatch::{Backend,DeviceBuffer,BinaryKind,UnaryKind};
    use std::sync::{Arc,Mutex};
    const DEV:Device=Device::Cuda(217);
    struct Buf(usize);
    impl DeviceBuffer for Buf {
        fn device(&self)->Device {DEV}
        fn len(&self)->usize {self.0}
        fn as_any(&self)->&dyn std::any::Any {self}
    }
    struct Sentinel;
    impl Backend for Sentinel {
        fn unary(&self,_:UnaryKind,_:&[f32])->Vec<f32>{panic!("unexpected compute")}
        fn binary(&self,_:BinaryKind,_:&[f32],_:&[f32])->Vec<f32>{panic!("unexpected compute")}
        fn matmul(&self,_:&[f32],_:&[f32],_:usize,_:usize,_:usize)->Vec<f32>{panic!("unexpected compute")}
        fn alloc_from_host(&self,x:&[f32])->Result<Box<dyn DeviceBuffer>>{Ok(Box::new(Buf(x.len())))}
        fn copy_to_host(&self,_:&dyn DeviceBuffer)->Result<Vec<f32>>{panic!("must reject before download")}
    }
    static LOCK:Mutex<()>=Mutex::new(());
    let _guard=LOCK.lock().unwrap_or_else(|e|e.into_inner());
    ferro_core::register_backend(DEV,Arc::new(Sentinel));
    let x=Tensor::full(&[1,1],0.5); let c=Tensor::ones(&[1,1,4]);
    let xd=x.to_device(DEV).unwrap(); let cd=c.to_device(DEV).unwrap();
    assert!(matches!(cubic().evaluate(&xd),Err(Error::Unsupported{..})));
    assert!(matches!(kan(&xd,&c,&cubic()),Err(Error::Unsupported{..})));
    assert!(matches!(kan(&x,&cd,&cubic()),Err(Error::Unsupported{..})));
    assert!(matches!(basis::KanLayer::from_coefficients(cubic(),cd),Err(Error::Unsupported{..})));
}


#[test]
fn repeated_knot_quadratics_match_piecewise_polynomial_oracle() {
    let b=BSplineBasis::new(vec![0.,0.,0.,0.5,0.5,1.,1.,1.],2,OutsideDomain::Zero).unwrap();
    let xv:Vec<f32>=(0..513).map(|i|i as f32/512.).collect();
    let x=Tensor::from_vec(xv.clone(), &[513]).unwrap().requires_grad_(true).unwrap();
    let values=b.evaluate(&x).unwrap();
    let weights=Tensor::from_vec(vec![0.4,-0.7,1.2,0.3,-0.6], &[5]).unwrap();
    values.mul(&weights).unwrap().sum().backward();
    let dx=x.grad().unwrap().to_vec(); let w=weights.to_vec();
    for (r,(&x,row)) in xv.iter().zip(values.to_vec().chunks(5)).enumerate() {
        let (u,start)=if x<0.5 {(2.*x as f64,0)}else{(2.*x as f64-1.,2)};
        let mut oracle=[0f64;5]; let mut derivative=[0f64;5];
        oracle[start]=(1.-u).powi(2); oracle[start+1]=2.*u*(1.-u); oracle[start+2]=u*u;
        derivative[start]=-4.*(1.-u); derivative[start+1]=4.-8.*u; derivative[start+2]=4.*u;
        for (&a,e) in row.iter().zip(oracle) { assert_eq!(a,e as f32); }
        let expected=derivative.iter().zip(&w).map(|(&d,&w)|d*w as f64).sum::<f64>() as f32;
        assert_eq!(dx[r],expected);
    }
}

#[test]
fn strided_inputs_coefficients_and_empty_batches() {
    let x=Tensor::from_vec(vec![0.1,0.3,0.5,0.2,0.4,0.6], &[2,3]).unwrap().transpose(0,1).unwrap();
    let c=Tensor::from_vec((0..24).map(|i|i as f32/24.).collect(), &[2,3,4]).unwrap().transpose(0,1).unwrap();
    let xc=Tensor::from_vec(x.to_vec(),x.shape()).unwrap();
    let cc=Tensor::from_vec(c.to_vec(),c.shape()).unwrap();
    assert_eq!(kan(&x,&c,&cubic()).unwrap().to_vec(),kan(&xc,&cc,&cubic()).unwrap().to_vec());
    assert_eq!(kan(&Tensor::zeros(&[0,2]),&c,&cubic()).unwrap().shape(),&[0,3]);
    ferro_core::testkit::grad_check(&[x.transpose(0,1).unwrap(),c.transpose(0,1).unwrap()], |ts| {
        let xt=ts[0].transpose(0,1).unwrap();
        let ct=ts[1].transpose(0,1).unwrap();
        kan(&xt,&ct,&cubic()).unwrap().sum()
    });
}

#[test]
fn quadratic_partition_endpoints_and_repeated_knots() {
    let b = BSplineBasis::new(vec![0.,0.,0.,0.5,0.5,1.,1.,1.], 2, OutsideDomain::Zero).unwrap();
    let x = Tensor::from_vec(vec![-0.1,0.,0.125,0.5,0.875,1.,1.1], &[7]).unwrap();
    let y = b.evaluate(&x).unwrap();
    assert_eq!(y.shape(), &[7,5]);
    for (i,row) in y.to_vec().chunks(5).enumerate() {
        assert!(row.iter().all(|v| v.is_finite() && *v >= 0.));
        let expected = if i == 0 || i == 6 {0.} else {1.};
        assert!((row.iter().sum::<f32>() - expected).abs() < 1e-6);
    }
    assert_eq!(&y.to_vec()[5..10], &[1.,0.,0.,0.,0.]);
    assert_eq!(&y.to_vec()[25..30], &[0.,0.,0.,0.,1.]);
}
