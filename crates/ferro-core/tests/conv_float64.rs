use ferro_core::{DType, Tensor};

struct Case {
    base_shape: &'static [usize], weight_shape: &'static [usize], transpose: bool,
    stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], groups: usize,
    input: Vec<f64>, weight: Vec<f64>, output_shape: &'static [usize],
    output: Vec<f64>, seed: Vec<f64>, dx: Vec<f64>, dw: Vec<f64>,
}
include!("fixtures/conv_f64_goldens.rs");

fn tensor(v: Vec<f64>, shape: &[usize]) -> Tensor {
    Tensor::from_vec_f64(v, shape).unwrap().requires_grad_(true).unwrap()
}
fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && e.is_finite() && (a-e).abs() <= tolerance * e.abs().max(1.), "element {i}: {a} != {e}");
    }
}

#[test]
fn actual_torch_f64_grouped_depthwise_strided_and_empty_goldens() {
    for case in torch_cases() {
        let base = tensor(case.input, case.base_shape);
        let x = if case.transpose { base.transpose(2, 3).unwrap().detach().requires_grad_(true).unwrap() } else { base };
        let w = tensor(case.weight, case.weight_shape);
        let y = x.conv2d_with_options(&w, case.stride, case.padding, case.dilation, case.groups).unwrap();
        assert_eq!(y.dtype(), DType::F64); assert_eq!(y.shape(), case.output_shape);
        close(&y.to_vec_f64(), &case.output, 2e-12);
        let seed = Tensor::from_vec_f64(case.seed, y.shape()).unwrap();
        let gradients = Tensor::vjp_many_wrt_optional(&[&y], &[&x, &w], &[&seed], false).unwrap();
        close(&gradients[0].as_ref().unwrap().to_vec_f64(), &case.dx, 2e-12);
        close(&gradients[1].as_ref().unwrap().to_vec_f64(), &case.dw, 2e-12);
        assert!(gradients.iter().all(|g| !g.as_ref().unwrap().requires_grad()));
    }
}

#[test]
fn double_grouped_adjoint_matches_central_finite_differences() {
    let xv: Vec<f64> = (0..18).map(|i| 0.1 + i as f64 * 0.03).collect();
    let wv: Vec<f64> = (0..8).map(|i| -0.2 + i as f64 * 0.07).collect();
    let x = tensor(xv.clone(), &[1,2,3,3]); let w = tensor(wv.clone(), &[2,1,2,2]);
    let run = |x: &Tensor, w: &Tensor| x.conv2d_with_options(w, [2,1], [1,0], [1,1], 2).unwrap();
    let y = run(&x,&w); let coefficients: Vec<f64> = (0..y.numel()).map(|i| 0.13+i as f64*0.05).collect();
    let seed = Tensor::from_vec_f64(coefficients.clone(), y.shape()).unwrap();
    let g = Tensor::vjp_many_wrt_optional(&[&y], &[&x,&w], &[&seed], false).unwrap();
    for slot in 0..2 {
        let expected = g[slot].as_ref().unwrap().to_vec_f64();
        for index in 0..expected.len() {
            let eval = |delta: f64| {
                let (mut a,mut b) = (xv.clone(),wv.clone());
                if slot==0 { a[index]+=delta; } else { b[index]+=delta; }
                run(&Tensor::from_vec_f64(a,x.shape()).unwrap(),&Tensor::from_vec_f64(b,w.shape()).unwrap())
                    .to_vec_f64().iter().zip(&coefficients).map(|(a,b)|a*b).sum::<f64>()
            };
            close(&[(eval(1e-6)-eval(-1e-6))/2e-6], &[expected[index]], 2e-9);
        }
    }
}

#[test]
fn saved_inputs_are_guarded_output_is_not_and_failed_query_publishes_no_slots() {
    for changed in 0..2 {
        let x=tensor(vec![0.2,0.3,0.4,0.5],&[1,1,2,2]);let w=tensor(vec![0.7],&[1,1,1,1]);
        let y=x.conv2d(&w,1,0).unwrap();let seed=Tensor::ones(y.shape()).to_dtype(DType::F64);
        let before=Tensor::vjp_many_wrt_optional(&[&y],&[&x,&w],&[&seed],false).unwrap();
        y.detach().copy_from(&Tensor::zeros(y.shape()).to_dtype(DType::F64)).unwrap();
        let after=Tensor::vjp_many_wrt_optional(&[&y],&[&x,&w],&[&seed],false).unwrap();
        for i in 0..2 {assert_eq!(before[i].as_ref().unwrap().to_vec_f64(),after[i].as_ref().unwrap().to_vec_f64());}
        let value=if changed==0{&x}else{&w};value.detach().copy_from(&Tensor::ones(value.shape()).to_dtype(DType::F64)).unwrap();
        assert!(Tensor::vjp_many_wrt_optional(&[&y],&[&x,&w],&[&seed],false).is_err());
        assert!(x.grad().is_none() && w.grad().is_none());
    }
}

#[test]
fn dtype_capacity_capture_forward_and_higher_boundaries_precede_computation() {
    let x=tensor(vec![0.2;4],&[1,1,2,2]);let w=tensor(vec![0.7],&[1,1,1,1]);
    assert!(x.conv2d(&w.to_dtype(DType::F32),1,0).is_err());
    assert!(ferro_core::capture::capture(||x.conv2d(&w,1,0)).is_err());
    {let primal=Tensor::ones(&[1]);let seed=Tensor::ones(&[1]);
     let _level=ferro_core::forward_ad::DualLevel::new(&[&primal],&[&seed]).unwrap();
     assert!(x.conv2d(&w,1,0).is_err());}
    let y=x.conv2d(&w,1,0).unwrap();let seed=Tensor::ones(y.shape()).to_dtype(DType::F64);
    assert!(Tensor::vjp_many_wrt_optional(&[&y],&[&x],&[&seed],true).is_err());
    let too_large=Tensor::scalar(1.).to_dtype(DType::F64).expand(&[1,1,1,isize::MAX as usize/8+1]).unwrap();
    assert!(too_large.conv2d(&w,1,0).is_err());
    assert!(x.conv2d_with_options(&w,[0,1],[0,0],[1,1],1).is_err());
    assert!(x.conv2d_with_options(&w,[1,1],[usize::MAX,0],[1,1],1).is_err());
    assert!(x.conv2d_with_options(&w,[1,1],[0,0],[1,1],0).is_err());
}

#[test]
fn padded_ieee_zero_products_are_real_and_f32_path_remains_typed() {
    let x=tensor(vec![1.],&[1,1,1,1]);let w=tensor(vec![f64::INFINITY;4],&[1,1,2,2]);
    let y=x.conv2d(&w,1,1).unwrap();assert!(y.to_vec_f64().iter().all(|v|v.is_nan()));
    let xf=Tensor::from_vec(vec![1.,2.,3.,4.],&[1,1,2,2]).unwrap();
    let wf=Tensor::from_vec(vec![2.],&[1,1,1,1]).unwrap();
    let yf=xf.conv2d(&wf,1,0).unwrap();assert_eq!(yf.dtype(),DType::F32);assert_eq!(yf.to_vec(),vec![2.,4.,6.,8.]);
}
