use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::DualLevel;
use ferro_core::ops_ext::div_rounding::DivisionRounding::{Floor, Trunc};

#[test]
fn floating_and_integer_values_and_safe_errors() {
    for dtype in [DType::F32, DType::F64, DType::I64] {
        let a = Tensor::from_vec_i64(vec![-7, -7, 7, 7], &[4]).unwrap().to_dtype(dtype);
        let b = Tensor::from_vec_i64(vec![3, -3, 3, -3], &[4]).unwrap().to_dtype(dtype);
        assert_eq!(a.div_rounding(&b, Floor).unwrap().to_dtype(DType::I64).to_vec_i64(), [-3,2,2,-3]);
        assert_eq!(a.div_rounding(&b, Trunc).unwrap().to_dtype(DType::I64).to_vec_i64(), [-2,2,2,-2]);
        assert_eq!(a.floor_divide(&b).unwrap().to_dtype(DType::I64).to_vec_i64(), [-3,2,2,-3]);
    }
    for (a,b) in [(1,0), (i64::MIN,-1)] {
        let a=Tensor::from_vec_i64(vec![a],&[]).unwrap(); let b=Tensor::from_vec_i64(vec![b],&[]).unwrap();
        if b.to_vec_i64()[0] == 0 { assert!(a.div_rounding(&b,Floor).is_err()); } else { assert_eq!(a.div_rounding(&b,Floor).unwrap().to_vec_i64(), [i64::MIN]); }
        assert!(a.div_rounding(&b,Trunc).is_err());
    }
    let yes=Tensor::from_vec_bool(vec![true],&[]).unwrap();
    assert!(yes.div_rounding(&yes,Floor).is_err());
    assert_eq!(yes.div_rounding(&Tensor::from_vec_i64(vec![2],&[]).unwrap(),Floor).unwrap().dtype(),DType::I64);
}

#[test]
fn zeros_are_detached_in_all_supported_derivative_modes() {
    for dtype in [DType::F32,DType::F64] { for mode in [Floor,Trunc] {
        let a=Tensor::from_vec_f64(vec![3.,f64::NAN,0.],&[3]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let b=Tensor::from_vec_f64(vec![2.,0.,0.],&[3]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let seed=Tensor::from_vec_f64(vec![f64::NAN,1.,f64::INFINITY],&[3]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let y=a.div_rounding(&b,mode).unwrap();
        for graph in [false,true] {
            if graph && dtype==DType::F64 { continue; }
            for g in y.vjp_wrt(&[&a,&b],&seed,graph).unwrap() { assert_eq!(g.dtype(),dtype); assert_eq!(g.to_dtype(DType::F64).to_vec_f64(),[0.,0.,0.]); assert!(!g.requires_grad()); }
        }
        y.sum().backward(); assert_eq!(a.grad().unwrap().to_dtype(DType::F64).to_vec_f64(),[0.,0.,0.]);
        if dtype==DType::F32 { for direction_dtype in [DType::F32,DType::F64] {
            let v=seed.to_dtype(direction_dtype).requires_grad_(true).unwrap();
            let level=DualLevel::new(&[&a,&b],&[&v,&v]).unwrap();
            let out=a.div_rounding(&b,mode).unwrap(); let tangent=level.tangent(&out).unwrap().unwrap();
            assert_eq!(tangent.dtype(),DType::F32); assert!(!tangent.requires_grad()); assert!(!tangent.is_structural_zero()); assert_eq!(tangent.to_vec(),[0.,0.,0.]);
        } }
    } }
}

#[test]
fn saved_divisor_and_conditionally_saved_dividend_match_original_edges() {
    for right_live in [false,true] { for which in 0..3 { for mode in [Floor,Trunc] {
        let a=Tensor::full(&[2],3.).requires_grad_(true).unwrap();
        let b=Tensor::full(&[2],2.).requires_grad_(right_live).unwrap();
        let y=a.div_rounding(&b,mode).unwrap();
        with_grad_enabled(false,|| match which {0=>a.fill_(4.),1=>b.fill_(4.),_=>y.detach().fill_(4.)}).unwrap();
        let result=y.sum().grad_wrt(&[&a],false);
        assert_eq!(result.is_err(),which==1||(which==0&&right_live));
    } } }
    for later_enable in [false,true] {
        let a=Tensor::ones(&[2]).requires_grad_(true).unwrap();
        let base=Tensor::ones(&[2]).requires_grad_(true).unwrap();
        let view=with_grad_enabled(false,||base.reshape(&[2])).unwrap();
        assert!(view.requires_grad() && view.is_leaf());
        let y=a.div_rounding(&view,Floor).unwrap();
        if later_enable { view.requires_grad_(true).unwrap(); }
        with_grad_enabled(false,||a.fill_(4.)).unwrap();
        assert_eq!(y.sum().grad_wrt(&[&a],false).unwrap()[0].to_vec(),[0.,0.]);
        assert!(view.grad().is_none() && base.grad().is_none());
    }
    let a=Tensor::full(&[2],3.).requires_grad_(true).unwrap();let b=Tensor::full(&[2],2.);
    let y=a.floor_divide(&b).unwrap();with_grad_enabled(false,||b.fill_(4.)).unwrap();
    assert!(format!("{}",y.sum().grad_wrt(&[&a],false).err().unwrap()).contains("not implemented"));
    let v=Tensor::full(&[2],1.); let level=DualLevel::new(&[&a],&[&v]).unwrap();
    let y=a.floor_divide(&b).unwrap();assert!(level.tangent(&y).is_err());
}

#[test]
fn strided_broadcast_layout_and_huge_empty_values_do_not_materialize_inputs() {
    let a=Tensor::from_vec((0..6).map(|v|v as f32+1.).collect(),&[2,3]).unwrap().transpose(0,1).unwrap();
    let b=Tensor::full(&[2,3],2.).transpose(0,1).unwrap();
    let y=a.div_rounding(&b,Floor).unwrap();assert_eq!(y.strides(),a.strides());assert_eq!(y.to_vec(),[0.,2.,1.,2.,1.,3.]);
    let empty=Tensor::from_vec(vec![],&[0,1usize<<60]).unwrap().requires_grad_(true).unwrap();
    let huge=Tensor::scalar(2.).expand(&[1,1usize<<60]).unwrap();
    assert_eq!(empty.div_rounding(&huge,Floor).unwrap().shape(),empty.shape());
    let empty_i=Tensor::from_vec_i64(vec![],&[0,1usize<<60]).unwrap();
    let zero=Tensor::from_vec_i64(vec![0],&[]).unwrap();assert_eq!(empty_i.div_rounding(&zero,Trunc).unwrap().numel(),0);
}

#[test]
fn zero_derivatives_preserve_cotangent_layout_and_result_layout_separately() {
    for dtype in [DType::F32,DType::F64] { for mode in [Floor,Trunc] { for strided_seed in [false,true] {
        let a=Tensor::full(&[3,2],3.).to_dtype(dtype).requires_grad_(true).unwrap();
        let b=Tensor::full(&[3,2],2.).to_dtype(dtype).requires_grad_(true).unwrap();
        let seed=if strided_seed {Tensor::full(&[2,3],f32::NAN).to_dtype(dtype).transpose(0,1).unwrap()} else {Tensor::full(&[3,2],f32::NAN).to_dtype(dtype)};
        let y=a.div_rounding(&b,mode).unwrap();
        for gradient in y.vjp_wrt_preserve(&[&a,&b],&seed).unwrap() {assert_eq!(gradient.strides(),seed.strides());assert!(!gradient.requires_grad());assert_eq!(gradient.to_dtype(DType::F64).to_vec_f64(),[0.;6]);}
        if dtype==DType::F32 {
            let level=DualLevel::new(&[&a],&[&seed.to_dtype(DType::F64)]).unwrap();
            let out=a.div_rounding(&b,mode).unwrap();let tangent=level.tangent(&out).unwrap().unwrap();
            assert_eq!(tangent.strides(),out.strides());assert_eq!(tangent.dtype(),out.dtype());
        }
    } } }
}

#[test]
fn adversarial_floors_and_validation_preserve_real_zero_and_immutable_inputs() {
    for dtype in [DType::F32,DType::F64] {
        let a=Tensor::from_vec_f64(vec![1.,-0.,0.,-1.,f64::INFINITY],&[5]).unwrap().to_dtype(dtype);
        let b=Tensor::from_vec_f64(vec![0.1,1.,-1.,0.,f64::INFINITY],&[5]).unwrap().to_dtype(dtype);
        let out=a.div_rounding(&b,Floor).unwrap().to_dtype(DType::F64).to_vec_f64();
        assert_eq!(out[0],9.);assert_eq!(out[1].to_bits(),(-0.0f64).to_bits());assert_eq!(out[2].to_bits(),(-0.0f64).to_bits());assert_eq!(out[3],f64::NEG_INFINITY);assert!(out[4].is_nan());
        let before=a._version();
        assert!(ferro_core::capture::capture(||a.div_rounding(&b,Floor)).is_err());
        assert_eq!(a._version(),before);assert!(!ferro_core::capture::is_recording());
        assert!(a.div_rounding(&Tensor::zeros(&[2]).to_dtype(dtype),Floor).is_err());
    }
    let x=Tensor::ones(&[2]).requires_grad_(true).unwrap();let y=x.floor_divide(&Tensor::ones(&[2])).unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(||y.sum().backward())).is_err());
    assert!(x.grad().is_none());
}
