//! Ownership-sensitive empty InputBuffer controls against Torch's measured
//! no-graph mutation versus graph-building Add policies.
use super::*;
const SHAPE: &[usize] = &[2,3,0,2];
const RETAINED: &[usize] = &[0,2,6,1];
const CANONICAL: &[usize] = &[6,2,2,1];

fn owned(dtype: DType, live: bool, offset: usize) -> Tensor {
    let allocation = if dtype == DType::F64 {
        Tensor::from_vec_f64(vec![7.;offset], &[offset]).unwrap()
    } else { Tensor::from_vec(vec![7.;offset], &[offset]).unwrap() };
    Tensor::from_parts(allocation.0.storage.clone(), SHAPE.to_vec(), RETAINED.to_vec(), offset, Device::Cpu, live, None)
}
fn zeros(dtype: DType, shape: &[usize]) -> Tensor { Tensor::zeros(shape).to_dtype(dtype) }

#[test]
fn unique_empty_input_buffer_retains_identity_flags_layout_and_advances_version() {
    for dtype in [DType::F32,DType::F64] { for live in [false,true] { for offset in [0,3] {
        let old=owned(dtype,live,offset);let id=old.id();let ptr=old._storage_ptr();let version=old.version();
        let result=crate::autograd::with_grad_enabled(false,||Tensor::accumulate_vjp(old,zeros(dtype,SHAPE))).unwrap();
        assert_eq!(result.id(),id);assert_eq!(result._storage_ptr(),ptr);assert_eq!(result.strides(),RETAINED);
        assert_eq!(result.0.offset,offset);assert_eq!(result.requires_grad(),live);assert_eq!(result.version(),version+1);
        assert!(!result.is_structural_zero());
        let values=result.0.storage.read();match &*values {
            Storage::F32(v)=>assert_eq!(v,&vec![7.;offset]),Storage::F64(v)=>assert_eq!(v,&vec![7.;offset]),_=>unreachable!(),
        }
    }}}
}

#[test]
fn borrowed_tensor_storage_and_graph_mode_keep_out_of_place_add() {
    for dtype in [DType::F32,DType::F64] {for live in [false,true] {
        for storage_alias in [false,true] {
            let old=owned(dtype,live,0);let id=old.id();let ptr=old._storage_ptr();
            let held=if storage_alias{old.detach()}else{old.clone()};
            let result=crate::autograd::with_grad_enabled(false,||Tensor::accumulate_vjp(old,zeros(dtype,SHAPE))).unwrap();
            assert_ne!(result.id(),id);assert_ne!(result._storage_ptr(),ptr);assert_eq!(result.strides(),CANONICAL);
            assert!(!result.requires_grad());assert_eq!(result.version(),0);assert_eq!(held.version(),0);assert_eq!(held.strides(),RETAINED);
        }
        let old=owned(dtype,live,0);let weak=Arc::downgrade(&old.0);let id=old.id();
        let result=crate::autograd::with_grad_enabled(true,||Tensor::accumulate_vjp(old,zeros(dtype,SHAPE))).unwrap();
        assert_ne!(result.id(),id);assert_eq!(result.strides(),CANONICAL);assert_eq!(result.version(),0);assert_eq!(result.requires_grad(),live);
        if live{assert_eq!(Tensor(weak.upgrade().unwrap()).version(),0);}
    }}
}

#[test]
fn invalid_metadata_and_immutable_zero_reject_before_version_changes() {
    for dtype in [DType::F32,DType::F64] {
        let old=owned(dtype,false,0);let other=if dtype==DType::F64{DType::F32}else{DType::F64};
        crate::autograd::with_grad_enabled(false,|| {
            assert!(crate::inplace::raw_empty_add_(&old,&zeros(other,SHAPE)).is_err());
            assert!(crate::inplace::raw_empty_add_(&old,&zeros(dtype,&[0])).is_err());assert_eq!(old.version(),0);
            let zero=Tensor::structural_zero(SHAPE,dtype,Device::Cpu).unwrap();
            assert!(crate::inplace::raw_empty_add_(&zero,&zeros(dtype,SHAPE)).is_err());
            let result=Tensor::accumulate_vjp(zero,zeros(dtype,SHAPE)).unwrap();assert_eq!(result.version(),0);assert!(!result.is_structural_zero());
        });
        assert!(crate::inplace::raw_empty_add_(&old,&zeros(dtype,SHAPE)).is_err());assert_eq!(old.version(),0);
    }
}

#[test]
fn active_forward_level_retains_real_add_and_output_tangent() {
    use crate::forward_ad::DualLevel;
    let x=owned(DType::F32,false,0);let direction=zeros(DType::F32,SHAPE);
    let level=DualLevel::functional(&[&x],&[&direction]).unwrap();
    let result=crate::autograd::with_grad_enabled(false,||Tensor::accumulate_vjp(x.clone(),zeros(DType::F32,SHAPE))).unwrap();
    assert_eq!(x.version(),0);assert_eq!(result.version(),0);assert!(level.tangent(&result).unwrap().is_some());drop(level);
}

#[test]
fn canonical_where_query_matches_unobserved_torch_empty_accumulation_modes() {
    for dtype in [DType::F32,DType::F64] {for graph in [false,true] {for permute in [false,true] {
        // General F64 graph-VJP remains outside the current native contract.
        if dtype==DType::F64 && graph {continue;}
        let x=zeros(dtype,&[2,0,3]).transpose(0,2).unwrap().requires_grad_(true).unwrap();
        let g=zeros(dtype,&[3,0,2]).requires_grad_(true).unwrap();let w=zeros(dtype,&[3,0,2]).requires_grad_(true).unwrap();
        let dg=zeros(dtype,SHAPE).requires_grad_(true).unwrap();let dw=zeros(dtype,SHAPE).requires_grad_(true).unwrap();
        let negative=dg.mul(&w).unwrap().add(&g.mul(&dw).unwrap()).unwrap();let condition=x.ge_tensor(&zeros(dtype,&[])).unwrap();
        let(condition,left,right)=if permute {
            (condition.permute(&[0,1,2]).unwrap(),dg.permute(&[0,1,2,3]).unwrap(),negative.permute(&[0,1,2,3]).unwrap())
        }else{(condition,dg.clone(),negative)};
        let y=Tensor::where_cond(&condition,&left,&right).unwrap();let y=if permute{y.permute(&[0,1,2,3]).unwrap()}else{y};
        let sum=y.sum();let seed=Tensor::scalar(1.).to_dtype(dtype);
        let gradients=Tensor::vjp_many_wrt_optional(&[&sum],&[&dg,&x],&[&seed],graph).unwrap();let result=gradients[0].as_ref().unwrap();
        assert_eq!(result.strides(),if !graph&&!permute{RETAINED}else{CANONICAL},"{dtype:?} graph{graph} permute{permute}");assert_eq!(result.version(),u64::from(!graph&&!permute));
        assert_eq!(result.requires_grad(),graph);assert!(gradients[1].is_none());assert!(x.grad().is_none()&&dg.grad().is_none());
    }}}
}
