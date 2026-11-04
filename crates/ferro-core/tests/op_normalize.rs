use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn normalize_values() {
    let a = Tensor::from_vec(vec![3.0, 4.0, 0.0, 0.0, 0.0, 2.0], &[2, 3]).unwrap();
    let got = a.normalize(1, 1e-8).unwrap();
    assert_eq!(got.shape(), &[2, 3]);
    let v = got.to_vec();
    let norm0 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let norm1 = (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]).sqrt();
    assert!((norm0 - 1.0).abs() < 1e-5);
    assert!((norm1 - 1.0).abs() < 1e-5);
    assert!((v[0] - 0.6).abs() < 1e-5);
    assert!((v[1] - 0.8).abs() < 1e-5);
    assert!((v[5] - 1.0).abs() < 1e-5);
}

#[test]
fn normalize_nonfinite_rows_match_torch_values_and_gradients() {
    let x = Tensor::from_vec(vec![f32::NAN, 1., f32::INFINITY, 1., f32::NEG_INFINITY, 0., 3., 4., 0., 0.], &[5, 2])
        .unwrap().requires_grad_(true).unwrap();
    let y = x.normalize(1, 1e-12).unwrap();
    let values = y.to_vec();
    assert!(values[0].is_nan() && values[1].is_nan());
    assert!(values[2].is_nan() && values[4].is_nan());
    assert_eq!([values[3], values[5]], [0., 0.]);
    assert_eq!(&values[6..], &[0.6, 0.8, 0., 0.]);
    y.sum().backward();
    let gradient = x.grad().unwrap().to_vec();
    assert!(gradient[..6].iter().all(|value| value.is_nan()));
    assert!((gradient[6] - 0.032).abs() < 1e-7);
    assert!((gradient[7] + 0.024).abs() < 1e-7);
    assert_eq!(&gradient[8..], &[1e12, 1e12]);
}

#[test]
fn normalize_virtual_empty_skips_unneeded_reduction_allocation() {
    let x = Tensor::zeros(&[1, 0]).expand(&[isize::MAX as usize / 4 + 1, 0]).unwrap().requires_grad_(true).unwrap();
    let y = x.normalize(1, 1e-12).unwrap();
    assert_eq!(y.shape(), x.shape());
    assert!(y.to_vec().is_empty());
    y.sum().backward();
    assert_eq!(x.grad().unwrap().shape(), x.shape());
    assert!(x.grad().unwrap().to_vec().is_empty());
    let huge = Tensor::scalar(1.).expand(&[isize::MAX as usize / 4 + 1]).unwrap();
    assert!(matches!(huge.normalize(0, 1e-12), Err(ferro_core::Error::InvalidShape { op: "normalize", .. })));
}

#[test]
fn normalize_out_of_range_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(a.normalize(1, 1e-8).is_err());
}

#[test]
fn normalize_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, -0.5, 0.7], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].normalize(1, 1e-8).unwrap().sum());
}

#[test]
fn normalize_zero_rows_have_finite_gradients_for_tiny_eps() {
    let seed = Tensor::from_vec(vec![2.0, -3.0, 4.0, -5.0], &[2, 2]).unwrap();
    for eps in [0.5, 1e-12, 1e-23, 1e-30] {
        let x = Tensor::zeros(&[2, 2]).requires_grad_(true).unwrap();
        let y = x.normalize(1, eps).unwrap();
        assert_eq!(y.to_vec(), vec![0.0; 4]);
        y.backward_with(&seed);
        assert_eq!(x.grad().unwrap().to_vec(), seed.to_vec().iter().map(|g| g / eps).collect::<Vec<_>>());
        assert!(x.grad().unwrap().to_vec().iter().all(|g| g.is_finite()));
    }
}

#[test]
fn normalize_tiny_nonzero_norm_avoids_backward_square_underflow() {
    let x = Tensor::from_vec(vec![1e-22, 1e-22], &[1, 2]).unwrap().requires_grad_(true).unwrap();
    let y = x.normalize(1, 1e-30).unwrap();
    for value in y.to_vec() { assert!((value - 0.713954746723175).abs() < 1e-7); }
    y.sum().backward();
    // PyTorch float32 reference includes subnormal rounding of sum(x*x).
    for value in x.grad().unwrap().to_vec() {
        assert!(value.is_finite());
        assert!((value / -1.3895518880279613e20 - 1.0).abs() < 2e-5, "{value}");
    }
}

#[test]
fn normalize_overflowed_norm_returns_zero_values_and_gradients() {
    for value in [1e20, 3e38] {
        let x = Tensor::full(&[2, 2], value).requires_grad_(true).unwrap();
        let y = x.normalize(1, 1e-12).unwrap();
        assert_eq!(y.to_vec(), vec![0.0; 4]);
        y.sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![0.0; 4]);
    }
}

#[test]
fn normalize_clamp_boundary_and_strided_gradients() {
    let x = Tensor::from_vec(vec![0.5, 0.0], &[2]).unwrap().requires_grad_(true).unwrap();
    x.normalize(0, 0.5).unwrap().backward_with(&Tensor::from_vec(vec![2.0, -3.0], &[2]).unwrap());
    assert_eq!(x.grad().unwrap().to_vec(), vec![0.0, -6.0]);
    let data = Tensor::from_vec(vec![0.1, 2.0, 0.2, -3.0, -0.1, 4.0], &[3, 2]).unwrap();
    grad_check(&[data], |t| t[0].transpose(0, 1).unwrap().normalize(1, 0.75).unwrap().sum());
}

#[test]
fn normalize_empty_shapes_and_invalid_dtype() {
    assert!(matches!(Tensor::zeros(&[2, 2]).normalize(1, f32::NAN),
        Err(ferro_core::Error::InvalidShape { op: "normalize", .. })));
    for (shape, dim) in [(vec![0, 3], 1), (vec![2, 0], 1), (vec![2, 0, 3], 2)] {
        let x = Tensor::zeros(&shape).requires_grad_(true).unwrap();
        let y = x.normalize(dim, 1e-12).unwrap();
        assert_eq!(y.shape(), shape);
        y.sum().backward();
        assert_eq!(x.grad().unwrap().shape(), shape);
        assert!(x.grad().unwrap().to_vec().is_empty());
    }
    for x in [Tensor::from_vec_i64(vec![], &[0]).unwrap(), Tensor::from_vec_f64(vec![], &[0]).unwrap()] {
        assert!(matches!(x.normalize(0, 1e-12), Err(ferro_core::Error::DtypeMismatch { op: "normalize", .. })));
    }
}

#[test]
fn normalize_preserves_stale_version_checks_and_rejects_capture() {
    use ferro_core::{capture, graph::CompiledChain, Error};
    let x = Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap().requires_grad_(true).unwrap();
    let y = x.normalize(0, 1e-12).unwrap();
    ferro_core::autograd::with_grad_enabled(false, || x.copy_leaf_from(&Tensor::ones(&[2]))).unwrap();
    assert!(y.grad_wrt(&[&x], false).is_err());
    assert!(x.grad().is_none());
    assert!(matches!(capture(|| x.normalize(0, 1e-12)), Err(Error::Unsupported { op: "normalize", .. })));
    let source = Tensor::scalar(2.0);
    let recorded = capture(|| source.mul(&source).unwrap());
    let compiled = CompiledChain::compile(&recorded).unwrap();
    source.copy_from(&Tensor::scalar(3.0)).unwrap();
    assert_eq!(compiled.replay().unwrap().item(), 9.0);
}
