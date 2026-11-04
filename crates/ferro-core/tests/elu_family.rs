use ferro_core::{DType, Tensor};
use ferro_core::ops_ext::add_alpha::Alpha;

fn values(values: &[f64], dtype: DType) -> Tensor {
    if dtype == DType::F64 { Tensor::from_vec_f64(values.to_vec(), &[values.len()]).unwrap() }
    else { Tensor::from_vec(values.iter().map(|x| *x as f32).collect(), &[values.len()]).unwrap() }
}
fn run(name: &str, x: &Tensor, alpha: Alpha) -> ferro_core::Result<Tensor> {
    match name { "elu" => x.elu_alpha(alpha), "celu" => x.celu_alpha(alpha), _ => x.selu() }
}

#[test]
fn signed_zeros_nan_branches_and_celu_reciprocal_stages() {
    for dtype in [DType::F32, DType::F64] {
        for name in ["elu", "celu", "selu"] {
            let x = values(&[-0., 0., f64::NAN], dtype).requires_grad_(true).unwrap();
            let y = run(name, &x, Alpha::F64(f64::INFINITY)).unwrap();
            assert_eq!(y.to_vec_f64()[0].to_bits(), (-0.0f64).to_bits());
            assert_eq!(y.to_vec_f64()[1].to_bits(), 0.0f64.to_bits());
            let g = y.sum().grad_wrt(&[&x], false).unwrap().remove(0).to_vec_f64();
            let scale = if name == "selu" { if dtype == DType::F32 { 1.0507009873554805f32 as f64 } else { 1.0507009873554805 } } else { 1. };
            assert_eq!(g[2], scale);
            if name == "celu" { assert!(g[0].is_nan() && g[1].is_nan()); }
        }
    }
    let x = values(&[0.], DType::F64).requires_grad_(true).unwrap();
    let y = x.celu_alpha(Alpha::F64(1e-40)).unwrap();
    assert_eq!(y.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec_f64(), vec![1.0000053899175751]);
    let x = values(&[0.], DType::F32).requires_grad_(true).unwrap();
    let y = x.celu_alpha(Alpha::F64(1e38)).unwrap();
    assert_eq!(y.vjp_wrt(&[&x], &values(&[0.3], DType::F32), false).unwrap()[0].to_vec(), vec![0.30000004172325134]);
}

#[test]
fn typed_layout_input_versions_and_output_mutation() {
    for dtype in [DType::F32, DType::F64] { for name in ["elu", "celu", "selu"] {
        let x = values(&[-0.2, 0.4, -1., 0.3, -0.7, 0.5], dtype).reshape(&[2,3]).unwrap().transpose(0,1).unwrap().requires_grad_(true).unwrap();
        let y = run(name, &x, Alpha::F64(0.7)).unwrap();
        assert_eq!(y.strides(), x.strides());
        let expected = y.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec_f64();
        y.transpose(0,1).unwrap().detach().copy_from(&values(&[9.;6], dtype).reshape(&[2,3]).unwrap()).unwrap();
        assert_eq!(y.sum().grad_wrt(&[&x], false).unwrap()[0].to_vec_f64(), expected);
        x.transpose(0,1).unwrap().detach().copy_from(&values(&[2.;6], dtype).reshape(&[2,3]).unwrap()).unwrap();
        assert!(y.sum().grad_wrt(&[&x], false).is_err());
        assert!(x.grad().is_none());
    } }
}

#[test]
fn coefficient_overflow_empty_metadata_and_explicit_derivative_boundary() {
    for name in ["elu", "celu", "selu"] {
        let empty = Tensor::zeros(&[1,0]).expand(&[usize::MAX,0]).unwrap().requires_grad_(true).unwrap();
        let y = run(name, &empty, Alpha::F64(1.)).unwrap();
        assert_eq!(y.shape(), empty.shape());
        assert_eq!(y.sum().grad_wrt(&[&empty], false).unwrap()[0].shape(), empty.shape());
        let huge = Tensor::zeros(&[1]).expand(&[isize::MAX as usize / 4 + 1]).unwrap();
        assert!(run(name, &huge, Alpha::F64(1.)).is_err());
        assert!(run(name, &Tensor::from_vec_i64(vec![1], &[1]).unwrap(), Alpha::F64(1.)).is_err());
        let x = Tensor::scalar(0.2).requires_grad_(true).unwrap();
        assert!(run(name, &x, Alpha::F64(1.)).unwrap().sum().grad_wrt(&[&x], true).is_err());
    }
    assert!(Tensor::zeros(&[0]).elu_alpha(Alpha::F64(1e100)).is_err());
    assert!(Tensor::zeros(&[0]).celu_alpha(Alpha::F64(1e-100)).is_err());
    let x = values(&[-1.,0.,1.], DType::F64).requires_grad_(true).unwrap();
    let y=x.celu_alpha(Alpha::F64(1e100)).unwrap();
    assert!(y.sum().grad_wrt(&[&x],false).is_err());
    let unused=values(&[1.,2.,3.],DType::F64).requires_grad_(true).unwrap();
    assert_eq!(y.sum().grad_wrt(&[&unused],false).unwrap()[0].to_vec_f64(),vec![0.;3]);
    let out = ferro_core::autograd::with_grad_enabled(false, || x.celu_alpha(Alpha::F64(1e100))).unwrap();
    assert_eq!(out.to_vec_f64(), vec![-1.,0.,1.]);
}

#[test]
fn f64_finite_difference_checks_all_three_activations() {
    for name in ["elu", "celu", "selu"] {
        let data = [-0.8,-0.3,0.4];
        let x = values(&data, DType::F64).requires_grad_(true).unwrap();
        let gradient = run(name,&x,Alpha::F64(2.)).unwrap().sum().grad_wrt(&[&x],false).unwrap()[0].to_vec_f64();
        for index in 0..data.len() {
            let mut plus=data;let mut minus=data;plus[index]+=1e-6;minus[index]-=1e-6;
            let p=run(name,&values(&plus,DType::F64),Alpha::F64(2.)).unwrap().to_vec_f64().iter().sum::<f64>();
            let m=run(name,&values(&minus,DType::F64),Alpha::F64(2.)).unwrap().to_vec_f64().iter().sum::<f64>();
            assert!(((p-m)/2e-6-gradient[index]).abs()<2e-9);
        }
    }
}

#[test]
fn celu_conversion_errors_are_deferred_pruned_and_leave_existing_slots_intact() {
    for count in [0,2] {
        let x=values(&vec![1.;count],DType::F64).requires_grad_(true).unwrap();
        let unused=values(&vec![2.;count],DType::F64).requires_grad_(true).unwrap();
        x.sum().backward();let old=x.grad().unwrap();let before=old.to_vec_f64();
        let output=x.celu_alpha(Alpha::F64(1e100)).unwrap();
        let seed=values(&vec![1.;count],DType::F64);
        assert!(output.vjp_wrt_optional(&[&unused],&seed,false).unwrap()[0].is_none());
        assert_eq!(output.vjp_wrt(&[&output],&seed,false).unwrap()[0].to_vec_f64(),seed.to_vec_f64());
        assert!(output.vjp_wrt(&[&x],&seed,false).is_err());
        assert_eq!(x.grad().unwrap().id(),old.id());assert_eq!(old.to_vec_f64(),before);
        let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||output.sum().backward()));
        assert!(result.is_err());
        assert_eq!(x.grad().unwrap().id(),old.id());assert_eq!(old.to_vec_f64(),before);
        assert!(unused.grad().is_none());
        assert!(ferro_core::autograd::is_grad_enabled());
    }
}

