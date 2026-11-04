use ferro_core::{dispatch::naive_matmul, DType, Tensor};

#[test]
fn reference_kernel_does_not_skip_zero_products_with_nonfinite_rhs() {
    let output = naive_matmul(&[0., 1., -0., 2.], &[f32::INFINITY, 3., f32::NAN, f32::NEG_INFINITY], 2, 2, 2);
    assert!(output[0].is_nan() && output[2].is_nan());
    assert_eq!(output[1], f32::NEG_INFINITY);
    assert_eq!(output[3], f32::NEG_INFINITY);
    assert_eq!(naive_matmul(&[0., 1., -0., 2.], &[7., 3., 5., -4.], 2, 2, 2), vec![5., -4., 10., -8.]);
}

#[test]
fn ordinary_zero_matmul_agrees_with_double_nonfinite_values() {
    for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let a = Tensor::zeros(&[2, 2]);
        let b = Tensor::from_vec(vec![value, 2., 3., value], &[2, 2]).unwrap();
        let actual = a.matmul(&b).unwrap().to_vec();
        let expected = a.to_dtype(DType::F64).matmul(&b.to_dtype(DType::F64)).unwrap().to_vec_f64();
        for (actual, expected) in actual.iter().zip(expected) { assert!(actual.is_nan() && expected.is_nan()); }
    }
}

#[test]
fn zero_cotangent_materializes_nan_against_infinite_saved_matrix() {
    let a = Tensor::ones(&[1, 1]).requires_grad_(true).unwrap();
    let b = Tensor::from_vec(vec![f32::INFINITY], &[1, 1]).unwrap().requires_grad_(true).unwrap();
    a.matmul(&b).unwrap().backward_with(&Tensor::zeros(&[1, 1]));
    assert!(a.grad().unwrap().to_vec()[0].is_nan());
    assert_eq!(b.grad().unwrap().to_vec(), vec![0.]);
}
