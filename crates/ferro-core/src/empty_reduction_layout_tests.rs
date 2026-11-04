use super::sum_empty_to_shape;
use crate::{DType, Tensor};
use crate::autograd::with_grad_enabled;

struct Golden {
    source: &'static [usize], target: &'static [usize], wide: bool, enabled: bool,
    stride: &'static [usize], identity: bool, live: bool, marker: bool,
    derivative_stride: Option<&'static [usize]>,
}
include!("../tests/fixtures/empty_reduction_layout_goldens.rs");

#[test]
fn twenty_actual_torch_empty_reductions_preserve_metadata_and_real_seed_graphs() {
    for (case,golden) in GOLDENS.iter().enumerate() {
        let g=if golden.wide {Tensor::from_vec_f64(Vec::new(),golden.source).unwrap()}else{Tensor::zeros(golden.source)}.requires_grad_(true).unwrap();
        let out=with_grad_enabled(golden.enabled,||sum_empty_to_shape(&g,golden.target)).unwrap();
        assert_eq!(out.shape(),golden.target,"case{case}");assert_eq!(out.strides(),golden.stride,"case{case}");
        assert_eq!(out.dtype(),if golden.wide {DType::F64}else{DType::F32});
        assert_eq!(out.id()==g.id(),golden.identity,"case{case} identity");
        assert_eq!(out.requires_grad(),golden.live,"case{case} recording");
        assert_eq!(out.is_structural_zero(),golden.marker,"case{case} marker");
        assert!(out.to_vec_f64().iter().all(|value|*value==0.));
        if let Some(strides)=golden.derivative_stride.filter(|_|!golden.wide) {
            let seed=Tensor::ones(golden.target).requires_grad_(true).unwrap();
            let derivative=Tensor::vjp_many_wrt_optional(&[&out],&[&g],&[&seed],true).unwrap().remove(0).unwrap();
            assert_eq!(derivative.shape(),g.shape());assert_eq!(derivative.strides(),strides,"case{case} higher strides");
            assert!(derivative.requires_grad());assert!(!derivative.is_structural_zero());
            assert!(derivative.depends_on(&seed));
        }
        assert!(g.grad().is_none());
    }
}

#[test]
fn empty_reduction_work_bound_keeps_virtual_results_and_huge_intermediates_safe() {
    for dtype in [DType::F32,DType::F64] {
        let empty=|shape:&[usize]|Tensor::from_vec_f64(Vec::new(),shape).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let g=empty(&[0,1usize<<60]);
        let out=sum_empty_to_shape(&g,&[1,1]).unwrap();
        assert_eq!(out.strides(),[1,1]);assert_eq!(out.storage_len(),1);
        let g=empty(&[1usize<<20,0]);
        let out=sum_empty_to_shape(&g,&[1usize<<20,1]).unwrap();
        assert_eq!(out.strides(),[1,1]);assert_eq!(out.storage_len(),1usize<<20);
        for size in [(1usize<<20)+1,1usize<<45] {
            let g=empty(&[size,0]);let out=sum_empty_to_shape(&g,&[size,1]).unwrap();
            assert_eq!(out.shape(),[size,1]);assert_eq!(out.storage_len(),1);
            assert!(out.requires_grad());assert!(!out.is_structural_zero());
            assert_eq!(out.narrow(0,size-1,1).unwrap().to_vec_f64(),[0.]);
        }
        let g=empty(&[2,1usize<<45,0]);
        let out=sum_empty_to_shape(&g,&[1usize<<45,0]).unwrap();
        assert_eq!(out.strides(),[1,1]);assert_eq!(out.storage_len(),0);
    }
}
