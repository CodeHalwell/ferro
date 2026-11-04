use ferro_core::{DType, Tensor};
use ferro_core::forward_ad::DualLevel;

fn source(dtype: DType) -> Tensor {
    Tensor::from_vec_f64(vec![0.2,0.4,0.3,0.7,0.1,0.6], &[2,3]).unwrap().to_dtype(dtype)
}

fn unary(input: &Tensor, index: usize) -> Tensor {
    match index { 0=>input.neg(), 1=>input.relu(), 2=>input.exp(), 3=>input.sigmoid(), 4=>input.tanh(), 5=>input.sqrt(), _=>input.sign().unwrap() }
}

#[test]
fn unary_dense_and_gapped_layouts_are_owned_without_changing_kernel_values() {
    for dtype in [DType::F32,DType::F64] {
        let base=source(dtype);
        let dense=base.transpose(0,1).unwrap();
        let gapped=Tensor::from_vec_f64((0..18).map(|v|v as f64/20.).collect(),&[2,9]).unwrap().to_dtype(dtype).slice(1,0,6,2).unwrap();
        let broadcast=base.narrow(0,0,1).unwrap().expand(&[2,3]).unwrap();
        for (input,stride) in [(&dense,vec![1,3]),(&gapped,vec![3,1]),(&broadcast,vec![3,1])] {
            for index in 0..7 {
                let out=unary(input,index);let contiguous=unary(&input.detach_copy(),index);
                assert_eq!(out.strides(),stride);assert_eq!(out.storage_offset(),0);
                assert_ne!(out._storage_ptr(),input._storage_ptr());
                assert_eq!(out.to_vec_f64(),contiguous.to_vec_f64());
            }
        }
        let output=dense.exp();let before=output.to_vec_f64();
        base.copy_from(&Tensor::zeros(&[2,3]).to_dtype(dtype)).unwrap();
        assert_eq!(output.to_vec_f64(),before);
    }
    let integers=Tensor::from_vec_i64(vec![i64::MIN,i64::MAX,0,-1,1,(1i64<<53)+1],&[2,3]).unwrap().transpose(0,1).unwrap();
    assert_eq!(integers.neg().strides(),&[1,3]);assert_eq!(integers.sign().unwrap().strides(),&[1,3]);
    assert_eq!(integers.neg().to_vec_i64(),integers.to_vec_i64().into_iter().map(i64::wrapping_neg).collect::<Vec<_>>());
}

#[test]
fn empty_offset_scalar_and_capacity_preflight_do_not_materialize_invalid_sources() {
    for dtype in [DType::F32,DType::F64,DType::I64] {
        let empty=Tensor::zeros(&[0,8]).to_dtype(dtype).narrow(1,4,4).unwrap().transpose(0,1).unwrap();
        let output=empty.sign().unwrap();assert_eq!(output.shape(),&[4,0]);assert_eq!(output.strides(),&[1,1]);
        assert_eq!(empty.storage_offset(),4);assert_eq!(output.storage_offset(),0);assert!(output.to_vec_f64().is_empty());
        let huge_empty=Tensor::zeros(&[1usize<<45,0]).to_dtype(dtype).transpose(0,1).unwrap().sign().unwrap();
        assert_eq!(huge_empty.numel(),0);assert_eq!(huge_empty.shape(),&[0,1usize<<45]);
        let scalar=Tensor::scalar(1.).to_dtype(dtype);assert_eq!(scalar.neg().shape(),&[]);
        let width=if dtype==DType::F32 {4} else {8};
        let huge=scalar.expand(&[isize::MAX as usize/width+1]).unwrap();
        assert!(huge.sign().is_err());
    }
}

#[test]
fn unary_direct_jvp_and_capture_keep_dense_layout_and_real_dependencies() {
    let base=source(DType::F32);
    let x=base.transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
    let v=Tensor::scalar(0.7).expand(x.shape()).unwrap().requires_grad_(true).unwrap();
    let level=DualLevel::new(&[&x],&[&v]).unwrap();let output=x.exp();
    let tangent=level.tangent(&output).unwrap().unwrap();drop(level);
    assert_eq!(output.strides(),&[1,3]);assert_eq!(tangent.strides(),&[1,3]);
    let gradients=tangent.sum().grad_wrt(&[&x,&v],true).unwrap();
    for (value,expected) in gradients[1].to_vec().into_iter().zip(output.to_vec()) { assert_eq!(value,expected); }
    assert!(gradients[0].requires_grad()&&gradients[1].requires_grad());assert!(x.grad().is_none()&&v.grad().is_none());
    let plain=source(DType::F32);let view=plain.transpose(0,1).unwrap();
    let captured=ferro_core::capture(||view.exp());let graph=ferro_core::graph::CompiledChain::compile(&captured).unwrap();
    plain.fill_(1.).unwrap();let replay=graph.replay().unwrap();
    assert_eq!(replay.strides(),&[1,3]);assert_eq!(replay.to_vec(),vec![1.0f32.exp();6]);
}

#[test]
fn saved_outputs_keep_ordinary_gradient_layout_and_output_only_version_guards() {
    for dtype in [DType::F32,DType::F64] {for index in 2..6 {
        let base=source(dtype);let x=base.transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
        let output=unary(&x,index);
        let seed=Tensor::scalar(1.).to_dtype(dtype).expand(output.shape()).unwrap();
        let query=|graph|Tensor::vjp_many_wrt_optional(&[&output],&[&x],&[&seed],graph).unwrap().remove(0).unwrap();
        let first=query(false);assert_eq!(first.strides(),&[1,3]);assert!(!first.requires_grad());
        assert_ne!(first._storage_ptr(),output._storage_ptr());assert_ne!(first._storage_ptr(),x._storage_ptr());
        let expected=output.to_vec_f64().into_iter().map(|y|if dtype==DType::F32 {
            let y=y as f32;(match index {2=>y,3=>(1.-y)*y,4=>(-y).mul_add(y,1.),_=>1./(2.*y)}) as f64
        }else{match index {2=>y,3=>(1.-y)*y,4=>(-y).mul_add(y,1.),_=>1./(2.*y)}}).collect::<Vec<_>>();
        assert_eq!(first.to_vec_f64(),expected);
        base.copy_from(&Tensor::ones(&[2,3]).to_dtype(dtype)).unwrap();
        assert_eq!(query(false).to_vec_f64(),expected);
        if dtype==DType::F32 {
            let recorded=query(true);assert_eq!(recorded.strides(),&[1,3]);assert_eq!(recorded.to_vec_f64(),expected);
            assert!(recorded.requires_grad());assert!(recorded.sum().grad_wrt(&[&x],false).is_ok());
        }
        output.transpose(0,1).unwrap().detach().copy_from(&Tensor::zeros(&[2,3]).to_dtype(dtype)).unwrap();
        assert!(Tensor::vjp_many_wrt_optional(&[&output],&[&x],&[&seed],false).is_err());
        assert!(x.grad().is_none());assert_eq!(first.to_vec_f64(),expected);
    }}
}

#[test]
fn backward_operand_layouts_preserve_actual_second_derivative_iteration_order() {
    for index in 2..6 {for seed_layout in 0..3 {
        let x=source(DType::F32).transpose(0,1).unwrap().detach().requires_grad_(true).unwrap();
        let seed=match seed_layout {0=>Tensor::scalar(0.7).expand(&[3,2]).unwrap(),1=>Tensor::full(&[2,3],0.7).transpose(0,1).unwrap(),_=>Tensor::full(&[3,2],0.7)}.requires_grad_(true).unwrap();
        let y=unary(&x,index);
        let first=Tensor::vjp_many_wrt_optional(&[&y],&[&x],&[&seed],true).unwrap().remove(0).unwrap();
        let incoming=Tensor::scalar(1.).expand(first.shape()).unwrap();
        let second=Tensor::vjp_many_wrt_optional(&[&first],&[&x,&seed],&[&incoming],false).unwrap();
        let expected=if index==4 || seed_layout==1 && index!=5 {[1,3]}else{[2,1]};
        assert_eq!(second[0].as_ref().unwrap().strides(),&expected);
        assert_eq!(second[1].as_ref().unwrap().strides(),&[1,3]);
        assert!(x.grad().is_none()&&seed.grad().is_none());
    }}
}
