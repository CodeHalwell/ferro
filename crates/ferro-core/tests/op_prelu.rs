use ferro_core::{DType, Tensor};

fn tensor(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(values.to_vec(), shape).unwrap().requires_grad_(true).unwrap()
}

#[test]
fn channel_weights_have_real_input_and_parameter_adjoint() {
    let x = tensor(&[-1., 2., -3., -4., 5., 6., 7., -8., 9., -10., -11., 12.], &[2, 3, 2]);
    let a = tensor(&[0.2, -0.3, 0.5], &[3]);
    ferro_core::testkit::grad_check(&[x.clone(), a.clone()], |args| args[0].prelu(&args[1]).unwrap().square().unwrap().sum());
    let y = x.prelu(&a).unwrap();
    y.sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), [0.2,1.,-0.3,-0.3,1.,1.,1.,0.2,1.,-0.3,0.5,1.]);
    assert_eq!(a.grad().unwrap().to_vec(), [-9., -17., -11.]);
}

#[test]
fn typed_precision_nonfinite_and_signed_zero_follow_each_branch() {
    let x = Tensor::from_vec_f64(vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN, -0.0, 0.0, -1e-300], &[6]).unwrap().requires_grad_(true).unwrap();
    let a = Tensor::from_vec_f64(vec![-0.25], &[]).unwrap().requires_grad_(true).unwrap();
    let y = x.prelu(&a).unwrap();
    let values = y.to_vec_f64();
    assert_eq!(values[0], f64::INFINITY); assert_eq!(values[1], f64::INFINITY); assert!(values[2].is_nan());
    assert_eq!(values[3].to_bits(), 0.0f64.to_bits()); assert_eq!(values[4].to_bits(), (-0.0f64).to_bits()); assert_eq!(values[5], 2.5e-301);
    let seed = Tensor::from_vec_f64(vec![1.;6], &[6]).unwrap();
    let grads = y.vjp_wrt(&[&x, &a], &seed, false).unwrap();
    assert_eq!(grads[0].to_vec_f64(), [1.,-0.25,-0.25,-0.25,-0.25,-0.25]);
    assert!(grads[1].to_vec_f64()[0].is_nan());
    let x = tensor(&[f32::INFINITY], &[1]);
    let a = tensor(&[f32::INFINITY], &[1]);
    let grads = x.prelu(&a).unwrap().vjp_wrt(&[&x,&a], &Tensor::from_vec(vec![f32::INFINITY], &[1]).unwrap(), false).unwrap();
    assert_eq!(grads[0].to_vec(), [f32::INFINITY]); assert_eq!(grads[1].to_vec(), [0.]);
}

#[test]
fn strided_inputs_scalar_and_empty_channels_retain_shapes() {
    let x = tensor(&[-1.,2.,-3.,4.,-5.,6.], &[2,3]).transpose(0,1).unwrap();
    let a = tensor(&[0.2,0.3], &[2]);
    let y = x.prelu(&a).unwrap();
    assert_eq!(y.shape(), &[3,2]); assert_eq!(y.to_vec(), [-0.2,4.,2.,-1.5,-0.6,6.]);
    y.sum().backward(); assert_eq!(a.grad().unwrap().to_vec(), [-4.,-5.]);
    let empty = tensor(&[], &[2,0,3]); let slope = tensor(&[], &[0]);
    let y = empty.prelu(&slope).unwrap(); y.sum().backward();
    assert_eq!(empty.grad().unwrap().shape(), &[2,0,3]); assert_eq!(slope.grad().unwrap().shape(), &[0]);
    let scalar = tensor(&[-2.], &[]).prelu(&tensor(&[0.3], &[1])).unwrap(); assert_eq!(scalar.shape(), &[]); assert_eq!(scalar.to_vec(), [-0.6]);
    for transposed in [false,true] {
        let values = Tensor::full(&[4,3,2], -2.);
        let source = if transposed { values.permute(&[2,1,0]).unwrap() } else { Tensor::full(&[2,3,4], -2.) };
        let source = source.requires_grad_(true).unwrap();
        let incoming = if transposed { Tensor::ones(&[2,3,4]) } else { Tensor::ones(&[4,3,2]).permute(&[2,1,0]).unwrap() };
        let output = source.prelu(&tensor(&[0.2,0.3,0.4], &[3])).unwrap();
        let gradient = Tensor::vjp_many_wrt_optional(&[&output], &[&source], &[&incoming], false).unwrap().remove(0).unwrap();
        assert_eq!(gradient.strides(), source.strides());
    }
    let width = 1usize << 40;
    let empty = Tensor::zeros(&[0,width,2]).requires_grad_(true).unwrap();
    let slope = Tensor::scalar(0.25).expand(&[width]).unwrap();
    let output = empty.prelu(&slope).unwrap();
    assert_eq!(output.shape(), &[0,width,2]);
    output.sum().backward();
    assert_eq!(empty.grad().unwrap().shape(), &[0,width,2]);
}

#[test]
fn validation_saved_versions_and_explicit_derivative_limits() {
    let x = tensor(&[1.,-2.,3.], &[3]); let a = tensor(&[0.2], &[1]);
    assert!(x.prelu(&Tensor::ones(&[3])).is_err());
    assert!(x.prelu(&Tensor::ones(&[1,1])).is_err());
    assert!(x.prelu(&Tensor::ones(&[1]).to_dtype(DType::F64)).is_err());
    let y = x.prelu(&a).unwrap();
    assert!(y.sum().grad_wrt(&[&x], true).is_err());
    for change in 0..3 {
        let x = tensor(&[1.,-2.,3.], &[3]); let a = tensor(&[0.2], &[1]); let y = x.prelu(&a).unwrap();
        [&x,&a,&y][change]._bump_version_for_test();
        assert_eq!(y.sum().grad_wrt(&[&x,&a], false).is_err(), change != 2);
    }
    let level = ferro_core::forward_ad::DualLevel::new(&[&x], &[&Tensor::ones(&[3])]).unwrap();
    let out = x.prelu(&a).unwrap(); assert!(level.tangent(&out).unwrap().is_some()); drop(level);
    assert!(ferro_core::capture::capture(|| x.prelu(&a)).is_err());
}
