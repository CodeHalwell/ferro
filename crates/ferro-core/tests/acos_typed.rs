use ferro_core::{DType, Tensor};
fn leaf(v: &[f64], shape: &[usize]) -> Tensor { Tensor::from_vec_f64(v.to_vec(), shape).unwrap().requires_grad_(true).unwrap() }
#[test]
fn typed_strided_values_and_finite_difference_gradient() {
    let x = leaf(&[-0.7, 0.2, 0.3, 0.8, -0.2, 0.1], &[2, 3]).transpose(0, 1).unwrap();
    let y = x.acos().unwrap(); assert_eq!(y.dtype(), DType::F64); assert_eq!(y.strides(), [1, 3]);
    let g = y.sum().grad_wrt(&[&x], false).unwrap().remove(0);
    for (actual, value) in g.to_vec_f64().iter().zip(x.to_vec_f64()) {
        let step=1e-6; let expected=((value+step).acos()-(value-step).acos())/(2.0*step);
        assert!((actual-expected).abs()<2e-10);
    }
}
#[test]
fn endpoint_seed_staging_and_saved_input_version() {
    let x=leaf(&[-1.,1.,0.,2.], &[4]); let out=x.acos().unwrap();
    let seed=leaf(&[0.,1.,-0.,1.], &[4]); let g=out.vjp_wrt(&[&x], &seed, false).unwrap().remove(0).to_vec_f64();
    assert!(g[0].is_nan()); assert_eq!(g[1], f64::NEG_INFINITY); assert!(g[2]==0. && !g[2].is_sign_negative()); assert!(g[3].is_nan());
    out._bump_version_for_test(); assert!(out.sum().grad_wrt(&[&x], false).is_ok());
    x._bump_version_for_test(); assert!(out.sum().grad_wrt(&[&x], false).is_err());
}
#[test]
fn huge_empty_layout_and_typed_capacity() {
    let x=leaf(&[], &[1,0]).expand(&[1_000_000_000_000,0]).unwrap(); let out=x.acos().unwrap();
    let grad=out.vjp_wrt_preserve(&[&x], &leaf(&[], out.shape())).unwrap().remove(0);
    assert_eq!(grad.shape(), x.shape());
    assert!(leaf(&[0.3], &[1]).expand(&[isize::MAX as usize/8+1]).unwrap().acos().is_err());
}
