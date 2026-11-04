use ferro_core::{DType, Tensor};

fn input(dtype: DType) -> Tensor {
    Tensor::from_vec_f64(vec![0.2,0.7,1.3,0.4,1.1,0.6],&[2,3]).unwrap().to_dtype(dtype).transpose(0,1).unwrap().detach().requires_grad_(true).unwrap()
}
fn output(x: &Tensor, kind: usize) -> Tensor {
    match kind {0=>x.expm1().unwrap(),1=>x.exp2().unwrap(),2=>x.reciprocal().unwrap(),_=>x.tan().unwrap()}
}
fn gradient(y: &Tensor, x: &Tensor, seed: &Tensor) -> ferro_core::Result<Tensor> {
    Tensor::vjp_many_wrt_optional(&[y],&[x],&[seed],false).map(|mut gradients|gradients.remove(0).unwrap())
}
fn assert_bits(actual: &[f64], expected: &[f64]) {
    for (&a,&e) in actual.iter().zip(expected) {assert!(a.is_nan()&&e.is_nan() || a.to_bits()==e.to_bits(),"{a:?} != {e:?}");}
}

#[test]
fn output_saved_adjoints_use_original_result_and_keep_physical_iteration_order() {
    for dtype in [DType::F32,DType::F64] {for kind in 0..4 {
        let x=input(dtype);let y=output(&x,kind);
        assert_eq!(y.strides(),&[1,3]);assert_ne!(y._storage_ptr(),x._storage_ptr());assert_eq!(y.storage_offset(),0);
        let seed=Tensor::from_vec_f64(vec![0.3],&[]).unwrap().to_dtype(dtype).expand(y.shape()).unwrap();
        let expected=y.to_vec_f64().into_iter().map(|v| if dtype==DType::F32 {
            let (v,g)=(v as f32,0.3f32);(match kind {0=>g*(v+1.),1=>(g*v)*std::f32::consts::LN_2,2=>(-g)*(v*v),_=>g*(v*v+1.)}) as f64
        }else{let g=0.3f64;match kind {0=>g*(v+1.),1=>(g*v)*std::f64::consts::LN_2,2=>(-g)*(v*v),_=>g*(v*v+1.)}}).collect::<Vec<_>>();
        let first=gradient(&y,&x,&seed).unwrap();assert_bits(&first.to_vec_f64(),&expected);
        assert_eq!(first.strides(),if kind==2 {&[2,1]}else{&[1,3]});assert!(!first.requires_grad());
        x.transpose(0,1).unwrap().detach().copy_from(&Tensor::full(&[2,3],9.).to_dtype(dtype)).unwrap();
        assert_bits(&gradient(&y,&x,&seed).unwrap().to_vec_f64(),&expected);
        y.transpose(0,1).unwrap().detach().copy_from(&Tensor::zeros(&[2,3]).to_dtype(dtype)).unwrap();
        assert!(gradient(&y,&x,&seed).is_err());assert!(x.grad().is_none());
        assert_bits(&first.to_vec_f64(),&expected);
    }}
}

#[test]
fn float64_input_saved_adjoints_preserve_layout_exact_values_and_version_guards() {
    let operations: &[fn(&Tensor)->Tensor]=&[
        |x|x.log(),|x|x.abs(),|x|x.sin().unwrap(),|x|x.cos().unwrap(),
        |x|x.sinh().unwrap(),|x|x.cosh().unwrap(),|x|x.log1p().unwrap(),|x|x.log10().unwrap(),
        |x|x.log2().unwrap(),|x|x.square().unwrap(),|x|x.powf64(3.3).unwrap(),|x|x.clamp_f64(0.3,1.).unwrap(),
        |x|x.gelu(),|x|x.silu(),
    ];
    for operation in operations {
        let x=input(DType::F64);let y=operation(&x);let seed=Tensor::ones(&[2,3]).to_dtype(DType::F64).transpose(0,1).unwrap();
        let first=gradient(&y,&x,&seed).unwrap();assert_eq!(y.strides(),&[1,3]);assert_eq!(first.strides(),&[1,3]);
        let values=first.to_vec_f64();y.transpose(0,1).unwrap().detach().copy_from(&Tensor::zeros(&[2,3]).to_dtype(DType::F64)).unwrap();
        assert_bits(&gradient(&y,&x,&seed).unwrap().to_vec_f64(),&values);
        x.transpose(0,1).unwrap().detach().copy_from(&Tensor::ones(&[2,3]).to_dtype(DType::F64)).unwrap();
        assert!(gradient(&y,&x,&seed).is_err());assert!(x.grad().is_none());
    }
}

#[test]
fn saved_expm1_cancellation_and_half_power_signed_zero_match_kernel_contracts() {
    for dtype in [DType::F32,DType::F64] {
        let x=Tensor::from_vec_f64(vec![-50.,-40.],&[2]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let y=x.expm1().unwrap();assert_eq!(y.to_vec_f64(),vec![-1.;2]);
        assert_bits(&gradient(&y,&x,&Tensor::ones(&[2]).to_dtype(dtype)).unwrap().to_vec_f64(),&[0.,0.]);
    }
    let x=Tensor::from_vec_f64(vec![f64::NEG_INFINITY,-0.0,0.0,f64::INFINITY,f64::NAN],&[5]).unwrap();
    assert_bits(&x.powf64(0.5).unwrap().to_vec_f64(),&[f64::NAN,-0.0,0.,f64::INFINITY,f64::NAN]);
    assert_bits(&x.powf64(-0.5).unwrap().to_vec_f64(),&[f64::NAN,f64::NEG_INFINITY,f64::INFINITY,0.,f64::NAN]);
    let x=Tensor::from_vec_f64(vec![2.],&[1]).unwrap().requires_grad_(true).unwrap();
    let seed=Tensor::from_vec_f64(vec![f64::from_bits(1)],&[1]).unwrap();
    assert_bits(&gradient(&x.powf64(3.).unwrap(),&x,&seed).unwrap().to_vec_f64(),&[f64::from_bits(12)]);
    let x=Tensor::from_vec_f64(vec![1e-150],&[1]).unwrap().requires_grad_(true).unwrap();
    let seed=Tensor::from_vec_f64(vec![1e308],&[1]).unwrap();
    assert_bits(&gradient(&x.powf64(3.).unwrap(),&x,&seed).unwrap().to_vec_f64(),&[1e308*(3.0*1e-150f64.powf(2.0))]);
}

#[test]
fn typed_unary_empty_views_and_capacity_errors_precede_materialization() {
    let source=Tensor::zeros(&[0,8]).to_dtype(DType::F64).narrow(1,4,4).unwrap().transpose(0,1).unwrap().requires_grad_(true).unwrap();
    for kind in 0..4 {
        let y=output(&source,kind);assert_eq!(y.shape(),&[4,0]);assert_eq!(y.strides(),&[1,1]);assert_eq!(y.storage_offset(),0);
        assert!(gradient(&y,&source,&Tensor::zeros(&[4,0]).to_dtype(DType::F64)).unwrap().to_vec_f64().is_empty());
    }
    let huge=Tensor::scalar(1.).to_dtype(DType::F64).expand(&[isize::MAX as usize/8+1]).unwrap();
    assert!(huge.log1p().is_err());assert!(huge.expm1().is_err());assert!(huge.powf64(1.).is_err());assert!(huge.clamp_f64(0.,1.).is_err());
    let empty=Tensor::zeros(&[1usize<<45,0]).to_dtype(DType::F64).transpose(0,1).unwrap();
    let y=empty.expm1().unwrap();assert_eq!(y.shape(),empty.shape());assert_eq!(y.numel(),0);
}
