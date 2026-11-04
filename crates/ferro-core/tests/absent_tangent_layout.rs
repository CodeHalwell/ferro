use ferro_core::{DType, Tensor};

#[test]
fn private_absent_zero_factory_preserves_layout_without_reading_or_recording() {
    for wide in [false,true] {
        let input=if wide {Tensor::from_vec_f64(vec![1.;6],&[2,3]).unwrap()}else{Tensor::ones(&[2,3])}.requires_grad_(true).unwrap().transpose(0,1).unwrap();
        let zero=input.zeros_like_layout().unwrap();
        assert_eq!(zero.dtype(),if wide {DType::F64}else{DType::F32});
        assert_eq!(zero.shape(),input.shape());assert_eq!(zero.strides(),input.strides());
        assert_eq!(zero.to_vec_f64(),vec![0.;6]);
        assert!(!zero.requires_grad()&&!zero.is_structural_zero());
        assert_ne!(input._storage_ptr(),zero._storage_ptr());
        let empty=input.narrow(0,0,0).unwrap().zeros_like_layout().unwrap();
        assert_eq!(empty.shape(),[0,2]);assert_eq!(empty.strides(),[1,3]);
        assert!(!empty.requires_grad()&&!empty.is_structural_zero());
    }
    let huge=Tensor::zeros(&[1usize<<45,0]);
    assert_eq!(huge.zeros_like_layout().unwrap().shape(),huge.shape());
    assert!(Tensor::scalar(1.).expand(&[1usize<<61]).unwrap().zeros_like_layout().is_err());
    assert!(Tensor::from_vec_i64(vec![1],&[1]).unwrap().zeros_like_layout().is_err());
}
