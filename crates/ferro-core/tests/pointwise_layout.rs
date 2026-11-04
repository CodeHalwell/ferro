use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::DualLevel;

fn values(dtype:DType)->Tensor { Tensor::from_vec_f64(vec![0.2,-0.4,0.3,0.7,-0.1,0.6],&[2,3]).unwrap().to_dtype(dtype) }

#[test]
fn dense_binary_outputs_preserve_physical_layout_without_aliasing_sources() {
    for dtype in [DType::F32,DType::F64] {
        let source=values(dtype);let a=source.transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
        let b=Tensor::full(&[3,1],2.).to_dtype(dtype).requires_grad_(true).unwrap();
        let y=a.mul(&b).unwrap().add(&a).unwrap();
        assert_eq!(y.strides(),&[1,3]);assert_eq!(y.storage_offset(),0);
        assert_ne!(y._storage_ptr(),a._storage_ptr());assert_ne!(y._storage_ptr(),b._storage_ptr());
        let g=y.sum().grad_wrt(&[&a,&b],false).unwrap();
        assert_eq!(g[0].to_vec_f64(),vec![3.;6]);
        for (actual,expected) in g[1].to_vec_f64().into_iter().zip([0.9,-0.5,0.9]) {assert!((actual-expected).abs()<1e-7);}
        let old=y.to_vec_f64();
        source.copy_from(&Tensor::zeros(&[2,3]).to_dtype(dtype)).unwrap();
        assert_eq!(y.to_vec_f64(),old);assert!(y.sum().grad_wrt(&[&a],false).is_err());
    }
}

#[test]
fn differentiable_copies_keep_dense_layout_and_leave_legacy_detached_cast_contiguous() {
    for before in [DType::F32,DType::F64,DType::I64,DType::Bool] {for after in [DType::F32,DType::F64,DType::I64,DType::Bool] {
        let base=Tensor::from_vec(vec![0.,1.,1.,0.,1.,0.],&[2,3]).unwrap().to_dtype(before);
        let source=base.transpose(0,1).unwrap();
        let live=matches!(before,DType::F32|DType::F64);let source=source.requires_grad_(live).unwrap();
        let copy=source.to_dtype_autograd(after,true).unwrap();
        assert_eq!(copy.strides(),&[1,3]);assert_ne!(copy._storage_ptr(),source._storage_ptr());
        assert_eq!(copy.to_vec_f64(),source.to_vec_f64());
        assert_eq!(copy.requires_grad(),live&&matches!(after,DType::F32|DType::F64));
        assert_eq!(source.to_dtype(after).strides(),&[2,1]);
        if before==after {assert_eq!(source.to_dtype_autograd(after,false).unwrap().id(),source.id());}
        if copy.requires_grad() {
            let gradient=copy.sum().grad_wrt(&[&source],false).unwrap().remove(0);
            assert_eq!(gradient.dtype(),before);assert_eq!(gradient.to_vec_f64(),vec![1.;6]);
            with_grad_enabled(false,||base.fill_typed_(match before {DType::F32=>ferro_core::inplace::FillValue::F32(1.),_=>ferro_core::inplace::FillValue::F64(1.)})).unwrap();
            assert!(copy.sum().grad_wrt(&[&source],false).is_ok());
        }
    }}
}

#[test]
fn strided_binary_jvp_and_higher_derivatives_keep_live_directions_and_old_slots() {
    let x=values(DType::F32).transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
    let v=Tensor::full(&[2,3],0.7).transpose(0,1).unwrap().requires_grad_(true).unwrap();
    let level=DualLevel::new(&[&x],&[&v]).unwrap();
    let y=x.mul(&x).unwrap();let dy=level.tangent(&y).unwrap().unwrap();drop(level);
    assert_eq!(y.strides(),&[1,3]);assert_eq!(dy.strides(),&[1,3]);
    let grad=dy.sum().grad_wrt(&[&x,&v],true).unwrap();
    assert_eq!(grad[0].to_vec(),vec![1.4;6]);
    for (actual,input) in grad[1].to_vec().into_iter().zip(x.to_vec()) {assert_eq!(actual,2.*input);}
    assert_eq!(grad[0].sum().grad_wrt(&[&v],false).unwrap()[0].to_vec(),vec![2.;6]);
    assert!(x.grad().is_none()&&v.grad().is_none());
    y.sum().backward();let held=x.grad().unwrap();assert_eq!(held.strides(),&[1,3]);
    y.sum().backward();assert_eq!(held.id(),x.grad().unwrap().id());
    assert_eq!(held.to_vec(),x.to_vec().into_iter().map(|v|4.*v).collect::<Vec<_>>());
    assert_eq!(x.grad().unwrap().strides(),&[1,3]);
}

#[test]
fn empty_offset_broadcast_and_capture_replay_keep_layout_without_large_reads() {
    for dtype in [DType::F32,DType::F64] {
        let x=Tensor::zeros(&[0,8]).to_dtype(dtype).narrow(1,4,4).unwrap().transpose(0,1).unwrap();
        let y=x.add(&x).unwrap();assert_eq!(y.shape(),&[4,0]);assert!(y.to_vec_f64().is_empty());
        assert_eq!(x.storage_offset(),4);
        let scalar=Tensor::scalar(1.).to_dtype(dtype);
        let huge=scalar.expand(&[1usize<<50,1]).unwrap();
        let empty=Tensor::zeros(&[1,0]).to_dtype(dtype);
        assert_eq!(huge.mul(&empty).unwrap().shape(),&[1usize<<50,0]);
    }
    let source=values(DType::F32);let x=source.transpose(0,1).unwrap();
    let output=ferro_core::capture(||x.mul(&x).unwrap().add(&x).unwrap());
    let replay=ferro_core::graph::CompiledChain::compile(&output).unwrap();
    source.fill_(2.).unwrap();let replayed=replay.replay().unwrap();
    assert_eq!(replayed.strides(),&[1,3]);assert_eq!(replayed.to_vec(),vec![6.;6]);
}
