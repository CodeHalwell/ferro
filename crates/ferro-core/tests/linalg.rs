use ferro_core::{linalg, DType, Tensor};
use ferro_core::testkit::grad_check_strict;

fn double(values: &[f64], shape: &[usize]) -> Tensor { Tensor::from_vec_f64(values.to_vec(), shape).unwrap() }
fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= tolerance * (1. + b.abs()), "{a} != {b}");
    }
}
fn grad64(values: &[f64], shape: &[usize], operation: impl Fn(&Tensor) -> Tensor) {
    let leaf = double(values, shape).requires_grad_(true).unwrap();
    operation(&leaf).sum().backward();
    let derivative = leaf.grad().unwrap();
    assert_eq!(derivative.dtype(), DType::F64);
    let mut expected = Vec::new();
    for index in 0..values.len() {
        let mut high = values.to_vec(); high[index] += 1e-6;
        let mut low = values.to_vec(); low[index] -= 1e-6;
        expected.push((operation(&double(&high, shape)).sum().to_vec_f64()[0]
            - operation(&double(&low, shape)).sum().to_vec_f64()[0]) / 2e-6);
    }
    close(&derivative.to_vec_f64(), &expected, 3e-8);
}


#[test]
fn cholesky_factor_solve_and_inverse_match_full_torch_factor_adjoints() {
    for upper in [false, true] {
        let a = double(&[2., 0.3, 0.5, 1.], &[2, 2]).requires_grad_(true).unwrap();
        let b = double(&[1., 2., 3., 4.], &[2, 2]).requires_grad_(true).unwrap();
        let out = linalg::cholesky_solve(&b, &a, upper).unwrap();
        close(&out.to_vec_f64(), if upper { &[-0.1775, -0.055, 2.85, 3.7] } else { &[-0.4375, -0.375, 2.75, 3.5] }, 2e-14);
        out.sum().backward();
        close(&a.grad().unwrap().to_vec_f64(), if upper { &[-0.0675, -4.55, -0.57626875, -11.437375] } else { &[0.3125, 0.24921875, -4.25, -9.309375] }, 2e-14);
        close(&b.grad().unwrap().to_vec_f64(), if upper { &[0.1225, 0.1225, 0.85, 0.85] } else { &[0.0625, 0.0625, 0.75, 0.75] }, 2e-14);
        let a = double(&[2., 0.3, 0.5, 1.], &[2, 2]).requires_grad_(true).unwrap();
        let out = linalg::cholesky_inverse(&a, upper).unwrap();
        close(&out.to_vec_f64(), if upper { &[0.2725, -0.15, -0.15, 1.] } else { &[0.3125, -0.25, -0.25, 1.] }, 2e-14);
        assert_eq!(out.to_vec_f64()[1], out.to_vec_f64()[2]);
        out.sum().backward();
        close(&a.grad().unwrap().to_vec_f64(), if upper { &[-0.1225, -0.85, -0.22325625, -1.549125] } else { &[-0.0625, -0.09609375, -0.75, -1.153125] }, 2e-14);
    }
}

#[test]
fn cholesky_factor_selected_tangents_and_rhs_match_finite_differences() {
    let av = [2., 0.3, -0.2, 1.1, 1.7, -0.1, 0.4, 2.1];
    let bv = [0.3, 0.5, 0.7, -0.4, 0.1, -0.2];
    for upper in [false, true] {
        grad64(&av, &[2, 1, 2, 2], |a| {
            let selected = if upper { a.triu(0).unwrap() } else { a.tril(0).unwrap() };
            linalg::cholesky_solve(&double(&bv, &[1, 3, 2, 1]), &selected, upper).unwrap()
        });
        grad64(&av, &[2, 1, 2, 2], |a| {
            let selected = if upper { a.triu(0).unwrap() } else { a.tril(0).unwrap() };
            linalg::cholesky_inverse(&selected, upper).unwrap()
        });
        grad64(&bv, &[1, 3, 2, 1], |b| linalg::cholesky_solve(b, &double(&av, &[2, 1, 2, 2]), upper).unwrap());
        let a = Tensor::from_vec(av[..4].iter().map(|&x| x as f32).collect(), &[2, 2]).unwrap();
        let b = Tensor::from_vec(vec![0.3, 0.7], &[2, 1]).unwrap();
        grad_check_strict(&[a, b], |inputs| {
            let selected = if upper { inputs[0].triu(0).unwrap() } else { inputs[0].tril(0).unwrap() };
            linalg::cholesky_solve(&inputs[1], &selected, upper).unwrap().sum()
        });
    }
}

#[test]
fn cholesky_factor_empty_capacity_ignored_nan_and_singular_boundaries() {
    let a = double(&[2., f64::NAN, 0.5, 1.], &[2, 2]).requires_grad_(true).unwrap();
    let b = double(&[1., 1.], &[2, 1]);
    let out = linalg::cholesky_solve(&b, &a, false).unwrap();
    close(&out.to_vec_f64(), &[0.0625, 0.75], 2e-14);
    out.sum().backward();
    let gradient = a.grad().unwrap().to_vec_f64();
    close(&[gradient[0], gradient[2]], &[-0.0625, -0.75], 2e-14);
    assert!(gradient[1].is_nan() && gradient[3].is_nan());
    for shape in [&[2, 0][..], &[0, 2, 1][..]] {
        let factor = double(&[2., f64::NAN, 0.5, 1.], &[2, 2]).requires_grad_(true).unwrap();
        linalg::cholesky_solve(&double(&[], shape), &factor, false).unwrap().sum().backward();
        let derivative = factor.grad().unwrap().to_vec_f64();
        if shape.len() == 2 {
            assert_eq!(derivative[0], 0.); assert_eq!(derivative[2], 0.);
            assert!(derivative[1].is_nan() && derivative[3].is_nan());
        } else { close(&derivative, &[0.; 4], 0.); }
    }
    let singular = double(&[0., 0., 0., 1.], &[2, 2]);
    assert!(linalg::cholesky_inverse(&singular, false).is_err());
    assert!(linalg::cholesky_solve(&b, &singular, false).unwrap().to_vec_f64().iter().any(|x| !x.is_finite()));
    let huge = double(&[], &[1usize << 60, 0, 0]).requires_grad_(true).unwrap();
    let out = linalg::cholesky_inverse(&huge, false).unwrap();
    out.sum().backward();
    assert_eq!(out.shape(), huge.shape());
    assert_eq!(huge.grad().unwrap().shape(), huge.shape());
    let a = double(&[2., 0., 0.5, 1.], &[2, 2]).requires_grad_(true).unwrap();
    let b = double(&[], &[1usize << 60, 2, 0]).requires_grad_(true).unwrap();
    let out = linalg::cholesky_solve(&b, &a, false).unwrap();
    out.sum().backward();
    assert_eq!(out.shape(), b.shape());
    close(&a.grad().unwrap().to_vec_f64(), &[0.; 4], 0.);
    assert!(linalg::cholesky_inverse(&double(&[], &[0, 1usize << 30, 1usize << 30]), false).is_err());
    assert!(linalg::cholesky_solve(&double(&[1., 2.], &[2]), &a, false).is_err());
    assert!(ferro_core::capture(|| linalg::cholesky_inverse(&a, false)).is_err());
}

#[test]
fn cholesky_factor_solve_inverse_saved_versions_and_higher_order_guards() {
    for inverse in [false, true] { for selected in 0..if inverse { 2 } else { 3 } {
        let a = double(&[2., 0.3, 0.5, 1.], &[2, 2]).requires_grad_(true).unwrap();
        let b = double(&[1., 2.], &[2, 1]).requires_grad_(true).unwrap();
        let out = if inverse { linalg::cholesky_inverse(&a, false) } else { linalg::cholesky_solve(&b, &a, false) }.unwrap();
        let candidates = if inverse { vec![&a, &out] } else { vec![&a, &b, &out] };
        candidates[selected]._bump_version_for_test();
        assert!(out.sum().grad_wrt(&[&a], false).is_err());
        assert!(a.grad().is_none());
    } }
    let a = Tensor::from_vec(vec![2., 0., 0.5, 1.], &[2, 2]).unwrap().requires_grad_(true).unwrap();
    assert!(linalg::cholesky_inverse(&a, false).unwrap().sum().grad_wrt(&[&a], true).is_err());
}

#[test]
fn pivoted_lu_left_right_inverse_values_and_precision() {
    let a = double(&[0., 2., 1., 3.], &[2, 2]);
    let b = double(&[4., 7.], &[2]);
    close(&linalg::solve(&a, &b, true).unwrap().to_vec_f64(), &[1., 2.], 1e-15);
    close(&linalg::solve(&a, &double(&[2., 8.], &[1, 2]), false).unwrap().to_vec_f64(), &[1., 2.], 1e-15);
    close(&linalg::inv(&a).unwrap().to_vec_f64(), &[-1.5, 1., 0.5, 0.], 1e-15);
    let tiny = double(&[1e-300], &[1, 1]);
    close(&linalg::solve(&tiny, &double(&[2e-300], &[1]), true).unwrap().to_vec_f64(), &[2.], 1e-15);
    let exact = 1. + 2f64.powi(-40);
    assert_eq!(linalg::inv(&double(&[exact], &[1, 1])).unwrap().to_vec_f64()[0], exact.recip());
    let batch = a.unsqueeze(0).unwrap().expand(&[2, 2, 2]).unwrap();
    assert_eq!(linalg::inv(&batch).unwrap().shape(), &[2, 2, 2]);
}

#[test]
fn solve_all_adjoint_inputs_and_broadcast_reduction_match_finite_differences() {
    let av = [2., 0.2, 0.1, 1.8, 1.7, -0.1, 0.3, 2.2];
    let bv = [0.4, 0.7, -0.2, 0.5, 0.3, -0.8];
    grad64(&av, &[2, 1, 2, 2], |a| linalg::solve(a, &double(&bv, &[1, 3, 2, 1]), true).unwrap());
    grad64(&bv, &[1, 3, 2, 1], |b| linalg::solve(&double(&av, &[2, 1, 2, 2]), b, true).unwrap());
    grad64(&av, &[2, 1, 2, 2], |a| linalg::solve(a, &double(&bv, &[1, 3, 1, 2]), false).unwrap());
    grad64(&bv, &[1, 3, 1, 2], |b| linalg::solve(&double(&av, &[2, 1, 2, 2]), b, false).unwrap());
    grad64(&av, &[2, 2, 2], |a| linalg::inv(a).unwrap());
    let a = Tensor::from_vec(av[..4].iter().map(|v| *v as f32).collect(), &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.4, 0.7], &[2]).unwrap();
    grad_check_strict(&[a, b], |inputs| linalg::solve(&inputs[0], &inputs[1], true).unwrap().sum());
}

#[test]
fn triangular_flags_and_ignored_entries_have_exact_zero_gradients() {
    let av = [3., 0.2, 0.7, 2.];
    let bv = [0.4, 0.8, -0.3, 0.2];
    for upper in [false, true] { for unit in [false, true] { for left in [false, true] {
        grad64(&av, &[2, 2], |a| linalg::solve_triangular(a, &double(&bv, &[2, 2]), upper, left, unit).unwrap());
        grad64(&bv, &[2, 2], |b| linalg::solve_triangular(&double(&av, &[2, 2]), b, upper, left, unit).unwrap());
    } } }
    let ignored = double(&[f64::NAN, 0.2, f64::NAN, f64::NAN], &[2, 2]).requires_grad_(true).unwrap();
    let result = linalg::solve_triangular(&ignored, &double(&[1., 2.], &[2, 1]), true, true, true).unwrap();
    close(&result.to_vec_f64(), &[0.6, 2.], 1e-15);
    result.sum().backward();
    close(&ignored.grad().unwrap().to_vec_f64(), &[0., -2., 0., 0.], 1e-15);
}

#[test]
fn cholesky_spd_symmetric_adjoint_finite_differences_and_failure_info() {
    let a = double(&[4., 2., 2., 3.], &[2, 2]);
    close(&linalg::cholesky(&a, false).unwrap().to_vec_f64(), &[2., 0., 1., 2f64.sqrt()], 1e-15);
    close(&linalg::cholesky(&a, true).unwrap().to_vec_f64(), &[2., 1., 0., 2f64.sqrt()], 1e-15);
    for upper in [false, true] {
        grad64(&[0.2, 0.1, -0.2, 0.4], &[2, 2], |x| {
            let matrix = x.add(&x.transpose(0, 1).unwrap()).unwrap().add(&double(&[3., 0., 0., 3.], &[2, 2])).unwrap();
            linalg::cholesky(&matrix, upper).unwrap()
        });
    }
    let mixed = double(&[4., 2., 2., 3., 1., 2., 2., 1.], &[2, 2, 2]);
    let (_, info) = linalg::cholesky_ex(&mixed, false, false).unwrap();
    assert_eq!(info.dtype(), DType::I64);
    assert_eq!(info.to_vec_i64(), vec![0, 2]);
    assert!(!info.requires_grad());
    assert!(linalg::cholesky_ex(&mixed, false, true).is_err());
}

#[test]
fn strided_inputs_empty_shapes_and_graph_guards() {
    let base = double(&[2., 0.1, 0.4, 1.8], &[2, 2]).requires_grad_(true).unwrap();
    let a = base.transpose(0, 1).unwrap();
    let b = double(&[1., 2., 3., 4.], &[2, 2]).transpose(0, 1).unwrap();
    let result = linalg::solve(&a, &b, true).unwrap();
    close(&a.matmul(&result).unwrap().to_vec_f64(), &b.to_vec_f64(), 1e-14);
    result.sum().backward();
    assert_eq!(base.grad().unwrap().shape(), base.shape());
    for shape in [&[0, 0][..], &[3, 0, 0][..], &[0, 2, 2][..]] {
        let empty = double(&[], shape).requires_grad_(true).unwrap();
        let inverse = linalg::inv(&empty).unwrap();
        assert_eq!(inverse.shape(), shape);
        inverse.sum().backward();
        assert_eq!(empty.grad().unwrap().shape(), shape);
        let (factor, info) = linalg::cholesky_ex(&empty, false, false).unwrap();
        assert_eq!(factor.shape(), shape);
        assert_eq!(info.shape(), &shape[..shape.len() - 2]);
    }
    assert!(ferro_core::capture(|| linalg::inv(&a)).is_err());
    assert!(linalg::inv(&base).unwrap().sum().grad_wrt(&[&base], true).is_err());
    let scalar = Tensor::from_vec(vec![2.], &[1, 1]).unwrap().requires_grad_(true).unwrap();
    assert!(linalg::inv(&scalar).unwrap().sum().grad_wrt(&[&scalar], true).is_err());
    let huge_empty = double(&[], &[1usize << 60, 2, 0]).requires_grad_(true).unwrap();
    let coefficient = double(&[2., 0., 0., 3.], &[2, 2]).requires_grad_(true).unwrap();
    let empty_solution = linalg::solve(&coefficient, &huge_empty, true).unwrap();
    assert_eq!(empty_solution.shape(), huge_empty.shape());
    empty_solution.sum().backward();
    assert_eq!(coefficient.grad().unwrap().to_vec_f64(), vec![0.; 4]);
    assert_eq!(huge_empty.grad().unwrap().shape(), huge_empty.shape());
}

#[test]
fn singular_invalid_metadata_capacity_and_stale_inputs_fail_cleanly() {
    assert!(linalg::inv(&double(&[1., 2., 2., 4.], &[2, 2])).is_err());
    assert!(linalg::solve(&double(&[0.; 4], &[2, 2]), &double(&[], &[0, 2, 1]), true).is_err());
    assert!(linalg::solve(&double(&[1.; 6], &[2, 3]), &double(&[1., 2.], &[2]), true).is_err());
    assert!(linalg::solve(&double(&[1.; 4], &[2, 2]), &Tensor::ones(&[2]), true).is_err());
    assert!(linalg::inv(&Tensor::from_vec_i64(vec![1], &[1, 1]).unwrap()).is_err());
    assert!(linalg::inv(&double(&[f64::NAN], &[1, 1])).is_err());
    let huge = double(&[1.], &[]).expand(&[isize::MAX as usize / 8 + 1, 1, 1]).unwrap();
    assert!(linalg::inv(&huge).is_err());
    let original = double(&[2., 0., 0., 3.], &[2, 2]);
    let leaf = original.requires_grad_(true).unwrap();
    let result = linalg::inv(&leaf).unwrap().sum();
    original._bump_version_for_test();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| result.backward())).is_err());
    assert!(leaf.grad().is_none());
}

#[test]
fn determinant_polynomial_singular_adjoints_and_logarithmic_extremes() {
    for data in [[1., 2., 2., 4.], [2., 0.2, -0.3, 3.], [0.; 4]] {
        grad64(&data, &[2, 2], |a| linalg::det(a).unwrap());
    }
    let singular = double(&[1., 2., 2., 4.], &[2, 2]).requires_grad_(true).unwrap();
    linalg::det(&singular).unwrap().backward();
    close(&singular.grad().unwrap().to_vec_f64(), &[4., -2., -2., 1.], 1e-15);
    grad64(&[1., 2., 3., 0., 2., 4., 1., 4., 7.], &[3, 3], |a| linalg::det(a).unwrap());
    grad64(&[2., 0.2, -0.3, 3.], &[2, 2], |a| linalg::slogdet(a).unwrap().1);
    for value in [1e-300, 1e300] {
        let matrix = double(&[value, 0., 0., value], &[2, 2]).requires_grad_(true).unwrap();
        let (sign, logarithm) = linalg::slogdet(&matrix).unwrap();
        assert_eq!(sign.to_vec_f64(), vec![1.]);
        close(&logarithm.to_vec_f64(), &[2. * value.ln()], 1e-15);
        linalg::det(&matrix).unwrap().backward();
        assert_eq!(matrix.grad().unwrap().to_vec_f64(), vec![value, 0., 0., value]);
    }
    let empty = double(&[], &[3, 0, 0]).requires_grad_(true).unwrap();
    assert_eq!(linalg::det(&empty).unwrap().to_vec_f64(), vec![1.; 3]);
    linalg::det(&empty).unwrap().sum().backward();
    assert_eq!(empty.grad().unwrap().shape(), &[3, 0, 0]);
    let zero = double(&[0.], &[1, 1]).requires_grad_(true).unwrap();
    linalg::det(&zero).unwrap().backward();
    assert_eq!(zero.grad().unwrap().to_vec_f64(), vec![1.]);
    let (sign, logarithm) = linalg::slogdet(&zero).unwrap();
    assert_eq!(sign.to_vec_f64(), vec![0.]);
    assert_eq!(logarithm.to_vec_f64(), vec![f64::NEG_INFINITY]);
    assert!(ferro_core::capture(|| linalg::det(&zero)).is_err());
    assert!(linalg::det(&zero).unwrap().grad_wrt(&[&zero], true).is_err());
}

#[test]
fn vector_norms_dimensions_ties_zero_empty_and_stable_extremes() {
    let data = [0.3, -1.2, 0.8, -0.7, 1.7, 0.4];
    for order in [0.5, 1., 2., 3., -1., -2., f64::INFINITY, f64::NEG_INFINITY] {
        for dims in [None, Some(&[0isize][..]), Some(&[-1isize][..])] {
            grad64(&data, &[2, 3], |a| linalg::vector_norm(a, order, dims, false).unwrap());
        }
    }
    let tied = double(&[2., -2., 1.], &[3]).requires_grad_(true).unwrap();
    linalg::vector_norm(&tied, f64::INFINITY, None, false).unwrap().backward();
    assert_eq!(tied.grad().unwrap().to_vec_f64(), vec![0.5, -0.5, 0.]);
    let tied_sums = double(&[1.1, 0.3, 0.4, 1.2], &[2, 2]).requires_grad_(true).unwrap();
    let columns = linalg::vector_norm(&tied_sums, 1., Some(&[0]), false).unwrap();
    assert_eq!(columns.to_vec_f64(), vec![1.5, 1.5]);
    linalg::vector_norm(&columns, f64::INFINITY, None, false).unwrap().backward();
    assert_eq!(tied_sums.grad().unwrap().to_vec_f64(), vec![0.5; 4]);
    let zero = double(&[0., 2.], &[2]).requires_grad_(true).unwrap();
    linalg::vector_norm(&zero, -1., None, false).unwrap().backward();
    assert_eq!(zero.grad().unwrap().to_vec_f64(), vec![0.; 2]);
    close(&linalg::vector_norm(&double(&[1e300, 1e300], &[2]), 2., None, false).unwrap().to_vec_f64(), &[2f64.sqrt() * 1e300], 1e-15);
    let tiny = linalg::vector_norm(&double(&[1e-300, 1e-300], &[2]), 2., None, false).unwrap().to_vec_f64()[0];
    assert!((tiny / 1e-300 - 2f64.sqrt()).abs() < 1e-15);
    let empty = double(&[], &[2, 0]).requires_grad_(true).unwrap();
    assert_eq!(linalg::vector_norm(&empty, 2., Some(&[-1]), true).unwrap().to_vec_f64(), vec![0.; 2]);
    linalg::vector_norm(&empty, 2., Some(&[-1]), true).unwrap().sum().backward();
    assert_eq!(empty.grad().unwrap().shape(), &[2, 0]);
    assert!(linalg::vector_norm(&empty, -1., Some(&[-1]), false).is_err());
    assert!(linalg::vector_norm(&zero, 2., Some(&[0, -1]), false).is_err());
    assert!(linalg::vector_norm(&zero, 2., Some(&[1]), false).is_err());
    assert!(ferro_core::capture(|| linalg::vector_norm(&zero, 2., None, false)).is_err());
    assert!(linalg::vector_norm(&zero, 2., None, false).unwrap().grad_wrt(&[&zero], true).is_err());
    let input = Tensor::from_vec(data.iter().map(|x| *x as f32).collect(), &[2, 3]).unwrap();
    grad_check_strict(&[input], |inputs| linalg::vector_norm(&inputs[0], 2., Some(&[0]), false).unwrap().sum());
}

#[test]
fn determinant_sign_and_zero_norm_preserve_unused_input_gradients() {
    for dtype in [DType::F32, DType::F64] {
        let a = double(&[2., 0., 0., -3.], &[2, 2]).to_dtype_autograd(dtype, false).unwrap().requires_grad_(true).unwrap();
        let sign = linalg::slogdet(&a).unwrap().0;
        sign.backward();
        assert!(a.grad().is_none());
        let zero_norm = linalg::vector_norm(&a, 0., None, false).unwrap();
        assert_eq!(zero_norm.to_vec(), vec![2.]);
        zero_norm.backward();
        assert!(a.grad().is_none());
        let gradient = zero_norm.grad_wrt(&[&a], false).unwrap();
        assert_eq!(gradient[0].dtype(), dtype);
        assert_eq!(gradient[0].to_vec(), vec![0.; 4]);
        assert!(a.grad().is_none());
        let live = linalg::det(&a).unwrap().add(&sign).unwrap();
        live.backward();
        assert_eq!(a.grad().unwrap().to_vec(), vec![-3., 0., 0., 2.]);
    }
}

#[test]
fn diagonal_views_alias_strides_offsets_gradients_empty_and_version_guards() {
    let original = double(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let view = linalg::diagonal(&original, 1, 0, 1).unwrap();
    assert_eq!(view.to_vec_f64(), vec![2., 6.]);
    original.copy_from(&double(&[7., 8., 9., 10., 11., 12.], &[2, 3])).unwrap();
    assert_eq!(view.to_vec_f64(), vec![8., 12.]);
    assert_eq!(linalg::trace(&original).unwrap().to_vec_f64(), vec![18.]);
    for offset in -4..=4 {
        grad64(&[0.2; 24], &[2, 3, 4], |a| linalg::diagonal(&a.transpose(0, 2).unwrap(), offset, 2, 0).unwrap());
    }
    let input = Tensor::from_vec(vec![0.2; 6], &[2, 3]).unwrap();
    grad_check_strict(&[input], |inputs| linalg::diagonal(&inputs[0], -1, 0, 1).unwrap().sum());
    let leaf = original.requires_grad_(true).unwrap();
    let result = linalg::diagonal(&leaf, 0, 0, 1).unwrap().sum();
    original._bump_version_for_test();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| result.backward())).is_err());
    assert!(leaf.grad().is_none());
    for shape in [&[2, 0, 3][..], &[1usize << 60, 0, 2][..]] {
        let empty = double(&[], shape).requires_grad_(true).unwrap();
        let result = linalg::diagonal(&empty, isize::MIN, 1, 2).unwrap();
        assert_eq!(result.shape(), &[shape[0], 0]);
        result.sum().backward();
        assert_eq!(empty.grad().unwrap().shape(), shape);
    }
    assert!(linalg::diagonal(&original, 0, 0, -2).is_err());
    assert!(linalg::diagonal(&original, 0, 0, 2).is_err());
    assert!(linalg::diagonal(&double(&[1.], &[]), 0, 0, 1).is_err());
    assert!(linalg::trace(&double(&[1., 2.], &[2])).is_err());
    assert!(ferro_core::capture(|| linalg::diagonal(&original, 0, 0, 1)).is_err());
    assert!(linalg::diagonal(&leaf, 0, 0, 1).unwrap().sum().grad_wrt(&[&leaf], true).is_err());
}

#[test]
fn reduced_qr_reconstruction_orthogonality_and_both_factor_adjoints() {
    let data = [2., 0.2, -0.1, 1.7, 0.4, -0.3, 0.5, 0.8];
    let a = double(&data, &[4, 2]);
    let (q, r) = linalg::qr(&a).unwrap();
    close(&q.matmul(&r).unwrap().to_vec_f64(), &data, 2e-14);
    close(&q.transpose(0, 1).unwrap().matmul(&q).unwrap().to_vec_f64(), &[1., 0., 0., 1.], 2e-14);
    assert_eq!(r.to_vec_f64()[2], 0.);
    grad64(&data, &[4, 2], |a| linalg::qr(a).unwrap().0);
    grad64(&data, &[4, 2], |a| linalg::qr(a).unwrap().1);
    grad64(&data, &[2, 2, 2], |a| { let (q, r) = linalg::qr(a).unwrap(); q.sum().add(&r.sum()).unwrap() });
    let input = Tensor::from_vec(data.iter().map(|&x| x as f32).collect(), &[4, 2]).unwrap();
    grad_check_strict(&[input], |inputs| { let (q, r) = linalg::qr(&inputs[0]).unwrap(); q.sum().add(&r.sum()).unwrap() });
}

#[test]
fn wide_qr_tail_and_leading_block_adjoints() {
    let data = [2., 0.2, 0.4, -0.1, 1.7, 0.3];
    let a = double(&data, &[2, 3]);
    let (q, r) = linalg::qr(&a).unwrap();
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 3]);
    close(&q.matmul(&r).unwrap().to_vec_f64(), &data, 2e-14);
    close(&q.transpose(0, 1).unwrap().matmul(&q).unwrap().to_vec_f64(), &[1., 0., 0., 1.], 2e-14);
    grad64(&data, &[2, 3], |a| linalg::qr(a).unwrap().0);
    grad64(&data, &[2, 3], |a| linalg::qr(a).unwrap().1);
    grad64(&data, &[1, 2, 3], |a| { let (q, r) = linalg::qr(a).unwrap(); q.sum().add(&r.sum()).unwrap() });
    let input = Tensor::from_vec(data.iter().map(|&x| x as f32).collect(), &[2, 3]).unwrap();
    grad_check_strict(&[input], |inputs| { let (q, r) = linalg::qr(&inputs[0]).unwrap(); q.sum().add(&r.sum()).unwrap() });
}

#[test]
fn qr_layout_empty_rank_and_graph_boundaries() {
    let data = double(&[2., 0.2, 0.1, 1.7, 0.4, -0.3], &[2, 3]).requires_grad_(true).unwrap();
    let matrix = data.transpose(0, 1).unwrap();
    let (q, r) = linalg::qr(&matrix).unwrap();
    close(&q.matmul(&r).unwrap().to_vec_f64(), &matrix.to_vec_f64(), 2e-14);
    q.sum().add(&r.sum()).unwrap().backward();
    assert_eq!(data.grad().unwrap().shape(), &[2, 3]);
    for shape in [&[3, 0][..], &[0, 3, 2][..], &[2, 0, 0][..], &[0, 3][..], &[1usize << 59, 0, 3][..]] {
        let input = double(&[], shape).requires_grad_(true).unwrap();
        let (q, r) = linalg::qr(&input).unwrap();
        q.sum().add(&r.sum()).unwrap().backward();
        assert_eq!(input.grad().unwrap().shape(), shape);
    }
    assert!(linalg::qr(&double(&[0.; 6], &[3, 2])).is_err());
    assert!(linalg::qr(&double(&[0.; 6], &[2, 3])).is_err());
    assert!(linalg::qr(&double(&[1., 0., f64::NAN, 0., 1., 2.], &[2, 3])).is_err());
    assert!(linalg::qr(&double(&[1., f64::NAN], &[2, 1])).is_err());
    assert!(ferro_core::capture(|| linalg::qr(&matrix)).is_err());
    assert!(linalg::qr(&matrix).unwrap().0.sum().grad_wrt(&[&data], true).is_err());
    let leaf = double(&[2., 0.2, 0.1, 1.7], &[2, 2]).requires_grad_(true).unwrap();
    let result = linalg::qr(&leaf).unwrap().1.sum();
    leaf._bump_version_for_test();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| result.backward())).is_err());
    assert!(leaf.grad().is_none());
}

#[test]
fn saved_output_and_sibling_versions_reject_before_any_gradient_publish() {
    for selected in 0..11 {
        let a = double(&[3., 0.2, 0.2, 2.], &[2, 2]).requires_grad_(true).unwrap();
        let (result, modified) = match selected {
            0 => { let out = linalg::inv(&a).unwrap(); (out.clone(), out) },
            1 => { let out = linalg::solve(&a, &double(&[1., 2.], &[2, 1]), true).unwrap(); (out.clone(), out) },
            2 => { let out = linalg::solve_triangular(&a, &double(&[1., 2.], &[2, 1]), true, true, false).unwrap(); (out.clone(), out) },
            3 => { let out = linalg::cholesky(&a, false).unwrap(); (out.clone(), out) },
            4 => { let out = linalg::det(&a).unwrap(); (out.clone(), out) },
            5 | 6 => { let out = linalg::vector_norm(&a, if selected == 5 { 2. } else { 0. }, None, false).unwrap(); (out.clone(), out) },
            7 => { let (q, r) = linalg::qr(&a).unwrap(); (q, r) },
            8 => { let (q, r) = linalg::qr(&a).unwrap(); (r, q) },
            9 => { let (sign, log) = linalg::slogdet(&a).unwrap(); (log, sign) },
            _ => { let out = linalg::slogdet(&a).unwrap().0; (out.clone(), out) },
        };
        modified._bump_version_for_test();
        let loss = result.sum();
        assert!(loss.grad_wrt(&[&a], false).is_err(), "case {selected}");
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loss.backward())).is_err(), "case {selected}");
        assert!(a.grad().is_none());
    }
    let a = double(&[3., 0.2, 0.2, 2.], &[2, 2]).requires_grad_(true).unwrap();
    let logarithm = linalg::slogdet(&a).unwrap().1;
    logarithm._bump_version_for_test();
    logarithm.backward();
    assert!(a.grad().is_some());
}

#[test]
fn symmetric_eigen_reconstruction_and_symmetric_manifold_adjoints() {
    let matrix = double(&[2., 0.3, 0.3, 4.], &[2, 2]);
    let (w, v) = linalg::eigh(&matrix, false).unwrap();
    close(&w.to_vec_f64(), &[3. - 1.09f64.sqrt(), 3. + 1.09f64.sqrt()], 2e-15);
    let diagonal = double(&[w.to_vec_f64()[0], 0., 0., w.to_vec_f64()[1]], &[2, 2]);
    close(&v.matmul(&diagonal).unwrap().matmul(&v.transpose(0, 1).unwrap()).unwrap().to_vec_f64(), &matrix.to_vec_f64(), 2e-15);
    for upper in [false, true] { for selected in 0..3 {
        grad64(&[1., 0.1, 0.2, 2.], &[2, 2], |x| {
            let matrix = x.add(&x.transpose(0, 1).unwrap()).unwrap();
            let (w, v) = linalg::eigh(&matrix, upper).unwrap();
            let w = w.mul(&double(&[0.3, -0.4], &[2])).unwrap();
            match selected { 0 => w, 1 => v, _ => w.sum().add(&v.sum()).unwrap() }
        });
    } }
    let input = Tensor::from_vec(vec![1., 0.1, 0.2, 2.], &[2, 2]).unwrap();
    grad_check_strict(&[input], |inputs| {
        let matrix = inputs[0].add(&inputs[0].transpose(0, 1).unwrap()).unwrap();
        let (w, v) = linalg::eigh(&matrix, false).unwrap();
        w.mul(&Tensor::from_vec(vec![0.3, -0.4], &[2]).unwrap()).unwrap().sum().add(&v.sum()).unwrap()
    });
}

#[test]
fn eigen_selected_triangle_scaling_repeats_and_empty_shapes() {
    let ignored = double(&[2., f64::NAN, 0.3, 4.], &[2, 2]).requires_grad_(true).unwrap();
    let (w, _) = linalg::eigh(&ignored, false).unwrap();
    w.sum().backward();
    close(&ignored.grad().unwrap().to_vec_f64(), &[1., 0., 0., 1.], 3e-15);
    assert!(linalg::eigh(&ignored, true).is_err());
    let tiny = double(&[1e308, 0., 0., 1e-308], &[2, 2]);
    assert_eq!(linalg::eigh(&tiny, false).unwrap().0.to_vec_f64(), vec![1e-308, 1e308]);
    let repeated = double(&[2., 0., 0., 2.], &[2, 2]).requires_grad_(true).unwrap();
    let (w, v) = linalg::eigh(&repeated, false).unwrap();
    w.sum().backward();
    close(&repeated.grad().unwrap().to_vec_f64(), &[1., 0., 0., 1.], 0.);
    v.sum().backward();
    assert!(repeated.grad().unwrap().to_vec_f64().iter().all(|x| x.is_nan()));
    for shape in [&[0, 0][..], &[0, 3, 3][..], &[1usize << 60, 0, 0][..]] {
        let input = double(&[], shape).requires_grad_(true).unwrap();
        let (w, v) = linalg::eigh(&input, false).unwrap();
        w.sum().add(&v.sum()).unwrap().backward();
        assert_eq!(input.grad().unwrap().shape(), shape);
    }
}

#[test]
fn eigen_capture_dtype_layout_and_saved_values_validate() {
    let a = double(&[2., 0.3, 0.3, 4.], &[2, 2]).requires_grad_(true).unwrap();
    let view = a.unsqueeze(0).unwrap().expand(&[2, 2, 2]).unwrap().transpose(1, 2).unwrap();
    let (w, v) = linalg::eigh(&view, true).unwrap();
    assert_eq!(w.shape(), &[2, 2]);
    assert_eq!(v.shape(), &[2, 2, 2]);
    assert!(ferro_core::capture(|| linalg::eigh(&a, false)).is_err());
    assert!(linalg::eigh(&a, false).unwrap().0.sum().grad_wrt(&[&a], true).is_err());
    assert!(linalg::eigh(&double(&[1., 2.], &[2]), false).is_err());
    assert!(linalg::eigh(&double(&[1.; 6], &[2, 3]), false).is_err());
    assert!(linalg::eigh(&Tensor::from_vec_i64(vec![1, 0, 0, 1], &[2, 2]).unwrap(), false).is_err());
    for selected in 0..4 {
        let (w, v) = linalg::eigh(&a, false).unwrap();
        if selected % 2 == 0 { w._bump_version_for_test(); } else { v._bump_version_for_test(); }
        let output = if selected < 2 { w } else { v };
        assert!(output.sum().grad_wrt(&[&a], false).is_err());
        assert!(a.grad().is_none());
    }
}

#[test]
fn svd_reconstruction_full_bases_and_all_factor_finite_differences() {
    let data = [2., 0.2, -0.1, 1.7, 0.4, -0.3];
    for shape in [[3, 2], [2, 3]] {
        let a = double(&data, &shape);
        for full in [false, true] {
            let (u, s, vh) = linalg::svd(&a, full).unwrap();
            let singular = s.to_vec_f64();
            assert!(singular[0] > singular[1] && singular[1] > 0.);
            let diagonal = double(&[singular[0], 0., 0., singular[1]], &[2, 2]);
            let thin_u = u.narrow(1, 0, 2).unwrap();
            let thin_vh = vh.narrow(0, 0, 2).unwrap();
            close(&thin_u.matmul(&diagonal).unwrap().matmul(&thin_vh).unwrap().to_vec_f64(), &data, 3e-14);
            let u_width = u.shape()[1];
            let v_width = vh.shape()[0];
            let identity_u: Vec<_> = (0..u_width * u_width).map(|i| if i / u_width == i % u_width { 1. } else { 0. }).collect();
            let identity_v: Vec<_> = (0..v_width * v_width).map(|i| if i / v_width == i % v_width { 1. } else { 0. }).collect();
            close(&u.transpose(0, 1).unwrap().matmul(&u).unwrap().to_vec_f64(), &identity_u, 3e-14);
            close(&vh.matmul(&vh.transpose(0, 1).unwrap()).unwrap().to_vec_f64(), &identity_v, 3e-14);
        }
        for selected in 0..4 {
            grad64(&data, &shape, |a| {
                let (u, s, vh) = linalg::svd(a, false).unwrap();
                let s = s.mul(&double(&[0.3, -0.4], &[2])).unwrap();
                match selected { 0 => u, 1 => s, 2 => vh, _ => u.sum().add(&s.sum()).unwrap().add(&vh.sum()).unwrap() }
            });
        }
    }
    let a = Tensor::from_vec(data.iter().map(|&x| x as f32).collect(), &[3, 2]).unwrap();
    grad_check_strict(&[a], |inputs| {
        let (u, s, vh) = linalg::svd(&inputs[0], false).unwrap();
        u.sum().add(&s.sum()).unwrap().add(&vh.sum()).unwrap()
    });
}

#[test]
fn svd_zero_rank_complements_empty_layouts_and_graph_guards() {
    for data in [[0.; 6], [1., 2., 2., 4., 0., 0.]] {
        let a = double(&data, &[3, 2]);
        let (u, s, vh) = linalg::svd(&a, true).unwrap();
        let singular = s.to_vec_f64();
        let diagonal = double(&[singular[0], 0., 0., singular[1]], &[2, 2]);
        close(&u.narrow(1, 0, 2).unwrap().matmul(&diagonal).unwrap().matmul(&vh).unwrap().to_vec_f64(), &data, 3e-14);
        close(&u.transpose(0, 1).unwrap().matmul(&u).unwrap().to_vec_f64(), &[1., 0., 0., 0., 1., 0., 0., 0., 1.], 3e-14);
    }
    let repeated = double(&[2., 0., 0., 2.], &[2, 2]).requires_grad_(true).unwrap();
    linalg::svd(&repeated, false).unwrap().1.sum().backward();
    close(&repeated.grad().unwrap().to_vec_f64(), &[1., 0., 0., 1.], 0.);
    let tall = double(&[2., 0.2, -0.1, 1.7, 0.4, -0.3], &[3, 2]).requires_grad_(true).unwrap();
    linalg::svd(&tall, true).unwrap().0.narrow(1, 2, 1).unwrap().sum().backward();
    close(&tall.grad().unwrap().to_vec_f64(), &[0.; 6], 0.);
    let wide = tall.transpose(0, 1).unwrap();
    assert_eq!(linalg::svd(&wide, true).unwrap().2.shape(), &[3, 3]);
    for shape in [&[0, 0][..], &[0, 3][..], &[3, 0][..], &[0, 3, 2][..], &[1usize << 60, 0, 0][..]] {
        for full in [false, true] {
            let a = double(&[], shape).requires_grad_(true).unwrap();
            let (u, s, vh) = linalg::svd(&a, full).unwrap();
            u.sum().add(&s.sum()).unwrap().add(&vh.sum()).unwrap().backward();
            assert_eq!(a.grad().unwrap().shape(), shape);
            if full && shape == [3, 0] { close(&u.to_vec_f64(), &[1., 0., 0., 0., 1., 0., 0., 0., 1.], 0.); }
            if full && shape == [0, 3] { close(&vh.to_vec_f64(), &[1., 0., 0., 0., 1., 0., 0., 0., 1.], 0.); }
        }
    }
    let huge = double(&[], &[1usize << 30, 0]);
    assert!(linalg::svd(&huge, true).is_err());
    assert!(linalg::svd(&double(&[f64::NAN], &[1, 1]), false).is_err());
    assert!(linalg::svd(&double(&[1.], &[1]), false).is_err());
    assert!(ferro_core::capture(|| linalg::svd(&tall, false)).is_err());
    assert!(linalg::svd(&tall, false).unwrap().1.sum().grad_wrt(&[&tall], true).is_err());
}

#[test]
fn svd_all_sibling_versions_validate_before_publishing_gradients() {
    for selected in 0..3 { for modified in 0..3 {
        let a = double(&[2., 0.2, -0.1, 1.7, 0.4, -0.3], &[3, 2]).requires_grad_(true).unwrap();
        let (u, s, vh) = linalg::svd(&a, false).unwrap();
        let outputs = [u, s, vh];
        outputs[modified]._bump_version_for_test();
        assert!(outputs[selected].sum().grad_wrt(&[&a], false).is_err());
        assert!(a.grad().is_none());
    } }
}

#[test]
fn pseudoinverse_full_rank_tall_wide_and_real_adjoints() {
    let data = [1.2, 0.3, -0.4, 1.7, 0.6, -0.2];
    for shape in [[3, 2], [2, 3]] {
        grad64(&data, &shape, |a| linalg::pinv(a, None, None, false).unwrap());
        let a = Tensor::from_vec(data.iter().map(|x| *x as f32).collect(), &shape).unwrap();
        grad_check_strict(&[a], |inputs| linalg::pinv(&inputs[0], None, None, false).unwrap().sum());
    }
    let a = double(&[2., 0.3, 0.3, 1.7], &[2, 2]);
    close(&linalg::pinv(&a, None, None, true).unwrap().to_vec_f64(), &linalg::inv(&a).unwrap().to_vec_f64(), 2e-14);
}

#[test]
fn pseudoinverse_rank_deficient_has_true_constant_rank_adjoint() {
    let a = double(&[1., 2., 2., 4., 0., 0.], &[3, 2]).requires_grad_(true).unwrap();
    let inverse = linalg::pinv(&a, None, None, false).unwrap();
    close(&inverse.to_vec_f64(), &[0.04, 0.08, 0., 0.08, 0.16, 0.], 2e-14);
    inverse.sum().backward();
    close(&a.grad().unwrap().to_vec_f64(), &[0.0048, -0.0144, -0.0144, -0.0768, 0.024, 0.048], 2e-14);
    assert_eq!(linalg::matrix_rank(&a, None, None, false).unwrap().to_vec_i64(), vec![1]);
    let zero = double(&[0.; 6], &[3, 2]).requires_grad_(true).unwrap();
    linalg::pinv(&zero, None, None, false).unwrap().sum().backward();
    close(&zero.grad().unwrap().to_vec_f64(), &[0.; 6], 0.);
}

#[test]
fn pseudoinverse_tolerance_batches_reduce_to_original_input() {
    let a = double(&[2., 0., 0., 0.5], &[2, 2]).requires_grad_(true).unwrap();
    let atol = double(&[0., 0.6, 2.], &[3]);
    let result = linalg::pinv(&a, Some(&atol), None, false).unwrap();
    assert_eq!(result.shape(), &[3, 2, 2]);
    close(&result.to_vec_f64(), &[0.5, 0., 0., 2., 0.5, 0., 0., 0., 0., 0., 0., 0.], 0.);
    assert_eq!(linalg::matrix_rank(&a, Some(&atol), None, false).unwrap().to_vec_i64(), vec![2, 1, 0]);
    result.sum().backward();
    close(&a.grad().unwrap().to_vec_f64(), &[-0.5, -0.75, -0.75, -4.], 2e-14);
    assert!(atol.grad().is_none());
}

#[test]
fn rank_and_inverse_tolerance_dtype_changes_exact_cutoff() {
    let a = Tensor::from_vec(vec![1., 0., 0., 0.1], &[2, 2]).unwrap();
    let narrow = Tensor::scalar(0.1);
    let wide = double(&[0.1000000009], &[]);
    assert_eq!(linalg::matrix_rank(&a, Some(&narrow), None, false).unwrap().to_vec_i64(), vec![1]);
    assert_eq!(linalg::matrix_rank(&a, Some(&wide), None, false).unwrap().to_vec_i64(), vec![2]);
    let small = (1.1f32 as f64 * 3. * f32::EPSILON as f64) as f32;
    let a = Tensor::from_vec(vec![1.1, 0., 0., 0., 0.1, 0., 0., 0., small], &[3, 3]).unwrap();
    let relative = Tensor::scalar(3. * f32::EPSILON);
    assert_eq!(linalg::matrix_rank(&a, None, None, false).unwrap().to_vec_i64(), vec![3]);
    assert_eq!(linalg::matrix_rank(&a, None, Some(&relative), false).unwrap().to_vec_i64(), vec![2]);
    let a = Tensor::from_vec(vec![33554432.], &[1, 1]).unwrap();
    let absolute = Tensor::from_vec_i64(vec![33554431], &[]).unwrap();
    assert_eq!(linalg::matrix_rank(&a, Some(&absolute), Some(&Tensor::scalar(0.)), false).unwrap().to_vec_i64(), vec![0]);
    assert_eq!(linalg::matrix_rank(&a, Some(&absolute), Some(&double(&[0.], &[])), false).unwrap().to_vec_i64(), vec![1]);
    let a = Tensor::from_vec(vec![((1u64 << 62) + (1u64 << 39)) as f32], &[1, 1]).unwrap();
    let absolute = Tensor::from_vec_i64(vec![(1i64 << 62) + (1i64 << 38) + 1], &[]).unwrap();
    assert_eq!(linalg::matrix_rank(&a, Some(&absolute), Some(&Tensor::scalar(0.)), false).unwrap().to_vec_i64(), vec![0]);
}

#[test]
fn pseudoinverse_empty_capacity_mutation_and_graph_guards() {
    let empty = double(&[], &[1usize << 60, 0, 0]).requires_grad_(true).unwrap();
    let inverse = linalg::pinv(&empty, None, None, false).unwrap();
    assert_eq!(inverse.shape(), empty.shape());
    inverse.sum().backward();
    assert_eq!(empty.grad().unwrap().shape(), empty.shape());
    assert!(linalg::matrix_rank(&empty, None, None, false).is_err());
    for selected in 0..2 {
        let a = double(&[2., 0.3, 0.1, 1.7], &[2, 2]).requires_grad_(true).unwrap();
        let atol = double(&[0.], &[]);
        let rtol = double(&[1e-10], &[]);
        let out = linalg::pinv(&a, Some(&atol), Some(&rtol), false).unwrap();
        [&a, &out][selected]._bump_version_for_test();
        assert!(out.sum().grad_wrt(&[&a], false).is_err());
        assert!(a.grad().is_none());
    }
    let a = double(&[2., 0.3, 0.1, 1.7], &[2, 2]).requires_grad_(true).unwrap();
    let tolerance = double(&[0.], &[]);
    let result = linalg::pinv(&a, Some(&tolerance), None, false).unwrap();
    tolerance._bump_version_for_test();
    assert!(result.sum().grad_wrt(&[&a], false).is_ok());
    let a = Tensor::ones(&[3, 2]).requires_grad_(true).unwrap();
    assert!(linalg::pinv(&a, None, None, true).is_err());
    let tolerance = Tensor::scalar(0.).requires_grad_(true).unwrap();
    assert!(linalg::pinv(&a, Some(&tolerance), None, false).is_err());
    assert!(!linalg::matrix_rank(&a, Some(&tolerance), None, false).unwrap().requires_grad());
    assert!(ferro_core::capture(|| linalg::pinv(&a, None, None, false)).is_err());
    assert!(linalg::pinv(&a, None, None, false).unwrap().sum().grad_wrt(&[&a], true).is_err());
}

#[test]
fn least_squares_tall_wide_both_input_adjoints() {
    for shape in [[3, 2], [2, 3]] {
        let data = [1.2, 0.3, -0.4, 1.7, 0.6, -0.2];
        let rhs: Vec<_> = (0..shape[0] * 2).map(|i| 0.2 + i as f64 * 0.1).collect();
        let b = double(&rhs, &[shape[0], 2]);
        grad64(&data, &shape, |a| linalg::lstsq(a, &b, None).unwrap().0);
        let a = double(&data, &shape);
        grad64(&rhs, &[shape[0], 2], |b| linalg::lstsq(&a, b, None).unwrap().0);
    }
    let a = Tensor::from_vec(vec![1.2, 0.3, -0.4, 1.7, 0.6, -0.2], &[3, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.2, 0.4, -0.7], &[3]).unwrap();
    grad_check_strict(&[a, b], |inputs| linalg::lstsq(&inputs[0], &inputs[1], None).unwrap().0.square().unwrap().sum());
}

#[test]
fn least_squares_batch_broadcast_gradients_and_metadata() {
    let a = Tensor::from_vec(vec![2., 0.3, 0.2, 1.7, 1.2, -0.2, 0.1, 1.4], &[2, 1, 2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.2, 0.4, -0.7, 0.3, 1., -0.4], &[1, 3, 2, 1]).unwrap();
    let result = linalg::lstsq(&a, &b, None).unwrap();
    assert_eq!(result.0.shape(), &[2, 3, 2, 1]);
    assert_eq!(result.1.shape(), &[0]);
    assert_eq!(result.2.shape(), &[2, 1]);
    assert_eq!(result.2.to_vec_i64(), vec![2, 2]);
    assert_eq!(result.3.shape(), &[0]);
    grad_check_strict(&[a, b], |inputs| linalg::lstsq(&inputs[0], &inputs[1], None).unwrap().0.square().unwrap().sum());
}

#[test]
fn least_squares_rank_deficiency_and_unused_residual_gradients() {
    let a = double(&[1., 0., 2., 0., 0., 0.], &[3, 2]).requires_grad_(true).unwrap();
    let b = double(&[1., 2., 3.], &[3]).requires_grad_(true).unwrap();
    let (solution, residuals, rank, singular_values) = linalg::lstsq(&a, &b, None).unwrap();
    close(&solution.to_vec_f64(), &[1., 0.], 2e-14);
    assert_eq!(rank.to_vec_i64(), vec![1]);
    assert!(residuals.requires_grad());
    assert!(!rank.requires_grad());
    assert!(!singular_values.requires_grad());
    residuals.sum().backward();
    assert!(a.grad().is_none()); assert!(b.grad().is_none());
    solution.sum().backward();
    close(&b.grad().unwrap().to_vec_f64(), &[0.2, 0.4, 0.], 2e-14);
}

#[test]
fn least_squares_empty_large_batches_and_original_version_contract() {
    let a = double(&[], &[1, 0, 0]).requires_grad_(true).unwrap();
    let b = double(&[], &[1usize << 60, 0, 0]).requires_grad_(true).unwrap();
    let (solution, _, rank, _) = linalg::lstsq(&a, &b, None).unwrap();
    assert_eq!(solution.shape(), &[1usize << 60, 0, 0]);
    assert_eq!(rank.shape(), &[1]);
    solution.sum().backward();
    assert_eq!(a.grad().unwrap().shape(), a.shape());
    assert_eq!(b.grad().unwrap().shape(), b.shape());
    assert!(linalg::lstsq(&b, &b, None).is_err());
    for selected in 0..3 { for residual in [false, true] {
        let a = double(&[2., 0.3, 0.1, 1.7], &[2, 2]).requires_grad_(true).unwrap();
        let b = double(&[0.2, 0.4], &[2, 1]).requires_grad_(true).unwrap();
        let (solution, residuals, _, _) = linalg::lstsq(&a, &b, None).unwrap();
        [&a, &b, &solution][selected]._bump_version_for_test();
        assert!(if residual { residuals } else { solution }.sum().grad_wrt(&[&a, &b], false).is_err());
        assert!(a.grad().is_none()); assert!(b.grad().is_none());
    } }
}

#[test]
fn least_squares_declared_scope_and_graph_boundaries() {
    let a = Tensor::ones(&[3, 2]).requires_grad_(true).unwrap();
    let b = Tensor::ones(&[3, 1]).requires_grad_(true).unwrap();
    for value in [0., -1., 1., 2., f64::INFINITY, f64::NAN] { assert!(linalg::lstsq(&a, &b, Some(value)).is_err()); }
    assert!(linalg::lstsq(&a, &double(&[1.; 3], &[3, 1]), None).is_err());
    assert!(linalg::lstsq(&a, &Tensor::ones(&[2, 1]), None).is_err());
    assert!(linalg::lstsq(&Tensor::ones(&[2, 3, 2]), &Tensor::ones(&[1, 3]), None).is_err());
    assert!(ferro_core::capture(|| linalg::lstsq(&a, &b, None)).is_err());
    assert!(linalg::lstsq(&a, &b, None).unwrap().0.sum().grad_wrt(&[&a, &b], true).is_err());
}

#[test]
fn least_squares_truncated_forward_uses_torch_default_cutoff_adjoint() {
    let a = double(&[2., 0., 0., 0.1], &[2, 2]).requires_grad_(true).unwrap();
    let b = double(&[1., 2.], &[2]).requires_grad_(true).unwrap();
    let result = linalg::lstsq(&a, &b, Some(0.2)).unwrap();
    close(&result.0.to_vec_f64(), &[0.5, 0.], 2e-14);
    assert_eq!(result.2.to_vec_i64(), vec![1]);
    result.0.sum().backward();
    close(&a.grad().unwrap().to_vec_f64(), &[-0.25, -10., -5., -200.], 2e-14);
    close(&b.grad().unwrap().to_vec_f64(), &[0.5, 10.], 2e-14);
    let plus = double(&[1., 2.00001], &[2]);
    let minus = double(&[1., 1.99999], &[2]);
    assert_eq!(linalg::lstsq(&a, &plus, Some(0.2)).unwrap().0.to_vec_f64(), linalg::lstsq(&a, &minus, Some(0.2)).unwrap().0.to_vec_f64());
}

#[test]
fn svd_deflates_stalled_subnormal_null_residue_without_losing_orthogonal_tiny_values() {
    let data = [-2.146925449371338, -1.6192692518234253, -0.42798247933387756,
        -1.439778208732605, 0.8278020620346069, -0.25986215472221375, 0., 0., 0.];
    let a = double(&data, &[3, 3]).requires_grad_(true).unwrap();
    let (u, s, vh) = linalg::svd(&a, false).unwrap();
    let singular = s.to_vec_f64();
    close(&singular, &[2.841636907950695, 1.4715081939137107, 0.], 3e-14);
    let diagonal = double(&[singular[0], 0., 0., 0., singular[1], 0., 0., 0., singular[2]], &[3, 3]);
    close(&u.matmul(&diagonal).unwrap().matmul(&vh).unwrap().to_vec_f64(), &data, 3e-14);
    close(&u.transpose(0, 1).unwrap().matmul(&u).unwrap().to_vec_f64(), &[1., 0., 0., 0., 1., 0., 0., 0., 1.], 3e-14);
    close(&vh.matmul(&vh.transpose(0, 1).unwrap()).unwrap().to_vec_f64(), &[1., 0., 0., 0., 1., 0., 0., 0., 1.], 3e-14);
    let b = double(&[0.2, -0.7, 1.3], &[3]).requires_grad_(true).unwrap();
    let (x, _, rank, _) = linalg::lstsq(&a, &b, None).unwrap();
    assert_eq!(rank.to_vec_i64(), vec![2]);
    close(&x.to_vec_f64(), &[0.2286170624180892, -0.43638923885914893, 0.03693582706923099], 3e-14);
    x.sum().backward();
    close(&a.grad().unwrap().to_vec_f64(), &[0.11822317524728516, -0.25644886973702424, 0.10385018155912039,
        0.028604809020130087, 0.03703788259896449, -0.24768334173458234,
        0.10908131461907139, 0.31838631632843156, 0.025095108563846252], 3e-14);
    close(&b.grad().unwrap().to_vec_f64(), &[-0.5849868380160229, 0.07691283109803448, 0.], 3e-14);
    let tiny = f64::from_bits(1);
    let a = double(&[1., 0., 0., tiny], &[2, 2]);
    assert_eq!(linalg::svd(&a, false).unwrap().1.to_vec_f64(), vec![1., tiny]);
}

#[test]
fn svd_rank_deficient_grid_retains_reconstruction_and_orthogonal_completed_bases() {
    for n in 2..9 { for null in 1..n.min(3) { for seed in 1..31 {
        let mut rng = seed as u64;
        let mut data = vec![0.; n * n];
        for row in 0..n - null { for column in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            data[row * n + column] = (rng >> 32) as i32 as f64 / i32::MAX as f64;
        } }
        let (u, s, vh) = linalg::svd(&double(&data, &[n, n]), false).unwrap();
        let singular = s.to_vec_f64();
        let mut diagonal = vec![0.; n * n];
        let mut identity = vec![0.; n * n];
        for i in 0..n { diagonal[i * n + i] = singular[i]; identity[i * n + i] = 1.; }
        close(&u.matmul(&double(&diagonal, &[n, n])).unwrap().matmul(&vh).unwrap().to_vec_f64(), &data, 6e-14);
        close(&u.transpose(0, 1).unwrap().matmul(&u).unwrap().to_vec_f64(), &identity, 6e-14);
        close(&vh.matmul(&vh.transpose(0, 1).unwrap()).unwrap().to_vec_f64(), &identity, 6e-14);
    } } }
}

#[test]
fn solve_ex_and_inv_ex_all_input_adjoints_match_finite_differences() {
    let a_values = [2., 0.3, 0.1, 1.7];
    let b_values = [0.2, -0.4, 0.7, 0.3];
    let a = double(&a_values, &[2, 2]);
    let b = double(&b_values, &[2, 2]);
    for left in [false, true] {
        grad64(&a_values, &[2, 2], |a| linalg::solve_ex(a, &b, left, false).unwrap().0);
        grad64(&b_values, &[2, 2], |b| linalg::solve_ex(&a, b, left, false).unwrap().0);
        let (x, info) = linalg::solve_ex(&a, &b, left, true).unwrap();
        close(&x.to_vec_f64(), &linalg::solve(&a, &b, left).unwrap().to_vec_f64(), 3e-14);
        assert_eq!(info.to_vec_i64(), vec![0]);
        assert!(!info.requires_grad());
    }
    grad64(&a_values, &[2, 2], |a| linalg::inv_ex(a, false).unwrap().0);
    let a = Tensor::from_vec(a_values.iter().map(|&x| x as f32).collect(), &[2, 2]).unwrap();
    let b = Tensor::from_vec(b_values.iter().map(|&x| x as f32).collect(), &[2, 2]).unwrap();
    grad_check_strict(&[a, b], |inputs| linalg::solve_ex(&inputs[0], &inputs[1], false, false).unwrap().0.square().unwrap().sum());
}

#[test]
fn solve_ex_status_is_first_original_matrix_pivot_and_not_rhs_broadcast_shape() {
    let a = double(&[2., 0.3, 0.1, 1.7, 0., 1., 0., 2.], &[2, 1, 2, 2]);
    let b = double(&[0.2; 12], &[1, 3, 2, 2]);
    for left in [false, true] {
        let (solution, info) = linalg::solve_ex(&a, &b, left, false).unwrap();
        assert_eq!(solution.shape(), &[2, 3, 2, 2]);
        assert_eq!(info.shape(), &[2, 1]);
        assert_eq!(info.dtype(), ferro_core::DType::I64);
        assert_eq!(info.to_vec_i64(), vec![0, 1]);
        assert!(linalg::solve_ex(&a, &b, left, true).is_err());
    }
    for (values, expected) in [([1., 2., 2., 4.], 2), ([0.; 4], 1), ([0., 1., 0., 2.], 1)] {
        let a = double(&values, &[2, 2]).requires_grad_(true).unwrap();
        let (inverse, info) = linalg::inv_ex(&a, false).unwrap();
        assert_eq!(info.to_vec_i64(), vec![expected]);
        assert!(inverse.to_vec_f64().iter().any(|value| !value.is_finite()));
        inverse.sum().backward();
        assert!(a.grad().is_some());
        assert!(linalg::inv_ex(&a, true).is_err());
    }
}

#[test]
fn ex_solve_empty_rhs_preserves_zero_gradients_and_status_capacity_preflight() {
    let a = double(&[2., 0., 0., 3.], &[1, 2, 2]).requires_grad_(true).unwrap();
    let b = double(&[], &[1usize << 60, 2, 0]).requires_grad_(true).unwrap();
    let (solution, info) = linalg::solve_ex(&a, &b, true, false).unwrap();
    assert_eq!(solution.shape(), b.shape());
    assert_eq!(info.shape(), &[1]);
    solution.sum().backward();
    close(&a.grad().unwrap().to_vec_f64(), &[0.; 4], 0.);
    assert_eq!(b.grad().unwrap().shape(), b.shape());
    let huge = double(&[], &[1usize << 60, 0, 0]);
    assert!(linalg::solve_ex(&huge, &huge, true, false).is_err());
    assert!(linalg::inv_ex(&huge, false).is_err());
    assert_eq!(linalg::solve(&huge, &huge, true).unwrap().shape(), huge.shape());
    let (inverse, info) = linalg::inv_ex(&double(&[], &[2, 0, 0]), false).unwrap();
    assert_eq!(inverse.shape(), &[2, 0, 0]);
    assert_eq!(info.to_vec_i64(), vec![0, 0]);
    assert!(ferro_core::capture(|| linalg::inv_ex(&a, false).map(|value| value.0)).is_err());
    assert!(linalg::inv_ex(&a, false).unwrap().0.sum().grad_wrt(&[&a], true).is_err());
}

#[test]
fn ex_solve_records_only_backward_value_dependencies() {
    for inverse in [false, true] { for selected in 0..4 {
        if inverse && selected == 1 { continue; }
        let a = double(&[2., 0.3, 0.1, 1.7], &[2, 2]).requires_grad_(true).unwrap();
        let b = double(&[0.2, 0.4], &[2]).requires_grad_(true).unwrap();
        let (output, info) = if inverse { linalg::inv_ex(&a, false).unwrap() } else { linalg::solve_ex(&a, &b, true, false).unwrap() };
        [&a, &b, &output, &info][selected]._bump_version_for_test();
        let result = output.sum().grad_wrt(&[&a, &b], false);
        if selected == 2 || (!inverse && selected == 0) { assert!(result.is_err()); }
        else { assert!(result.is_ok()); }
        assert!(a.grad().is_none()); assert!(b.grad().is_none());
    } }
}

#[test]
fn rectangular_lu_factor_and_unpacked_adjoints_match_finite_differences() {
    let data = [0.2, 2.1, 1.1, 0.3, 0.7, -0.4];
    for shape in [[3, 2], [2, 3]] {
        grad64(&data, &shape, |a| linalg::lu_factor(a).unwrap().0);
        grad64(&data, &shape, |a| { let (_, l, u) = linalg::lu(a).unwrap(); l.sum().add(&u.square().unwrap().sum()).unwrap() });
        let a = double(&data, &shape);
        let (p, l, u) = linalg::lu(&a).unwrap();
        close(&p.matmul(&l).unwrap().matmul(&u).unwrap().to_vec_f64(), &data, 3e-14);
        assert!(!p.requires_grad());
    }
    let a = Tensor::from_vec(data.iter().map(|&value| value as f32).collect(), &[3, 2]).unwrap();
    grad_check_strict(&[a], |inputs| linalg::lu_factor(&inputs[0]).unwrap().0.square().unwrap().sum());
}

#[test]
fn lu_solve_independent_packed_factor_and_rhs_adjoints_all_directions() {
    let a = double(&[0.2, 2.1, 1.1, 0.3], &[2, 2]);
    let (factor, pivots) = linalg::lu_factor(&a).unwrap();
    let rhs = [0.2, -0.4, 0.7, 0.3];
    let b = double(&rhs, &[2, 2]);
    assert_eq!(pivots.to_vec_i64(), vec![2, 2]);
    for left in [false, true] { for adjoint in [false, true] {
        grad64(&factor.to_vec_f64(), &[2, 2], |packed| linalg::lu_solve(packed, &pivots, &b, left, adjoint).unwrap());
        grad64(&rhs, &[2, 2], |b| linalg::lu_solve(&factor, &pivots, b, left, adjoint).unwrap());
        let coefficient = if adjoint { a.transpose(0, 1).unwrap() } else { a.clone() };
        close(&linalg::lu_solve(&factor, &pivots, &b, left, adjoint).unwrap().to_vec_f64(), &linalg::solve(&coefficient, &b, left).unwrap().to_vec_f64(), 3e-14);
    } }
    let a = double(&[0.2, 2.1, 1.1, 0.3, 2.3, 0.4, 0.2, 1.7], &[2, 1, 2, 2]);
    let (factor, pivots) = linalg::lu_factor(&a).unwrap();
    let factor = factor.requires_grad_(true).unwrap();
    let b = double(&[0.3; 6], &[1, 3, 2, 1]).requires_grad_(true).unwrap();
    linalg::lu_solve(&factor, &pivots, &b, true, false).unwrap().sum().backward();
    assert_eq!(factor.grad().unwrap().shape(), factor.shape());
    assert_eq!(b.grad().unwrap().shape(), b.shape());
}

#[test]
fn lu_metadata_status_empty_capacity_and_pivot_validation() {
    let a = double(&[0., 1., 0., 2.], &[2, 2]);
    let (packed, pivots, info) = linalg::lu_factor_ex(&a, false).unwrap();
    assert_eq!(info.to_vec_i64(), vec![1]);
    assert_eq!(pivots.to_vec_i64(), vec![1, 2]);
    assert!(linalg::lu_factor(&a).is_err());
    assert!(linalg::lu_factor_ex(&a, true).is_err());
    assert!(linalg::lu(&a).is_ok());
    for values in [vec![0, 2], vec![1, 3], vec![i64::MIN, 2]] {
        let bad = Tensor::from_vec_i64(values, &[2]).unwrap();
        assert!(linalg::lu_solve(&packed, &bad, &double(&[0.2, 0.3], &[2, 1]), true, false).is_err());
    }
    assert!(linalg::lu_solve(&packed, &pivots, &double(&[0.2, 0.3], &[2]), true, false).is_err());
    for shape in [&[0, 0][..], &[2, 0][..], &[0, 2][..], &[0, 2, 2][..]] {
        let a = double(&[], shape).requires_grad_(true).unwrap();
        let (p, l, u) = linalg::lu(&a).unwrap();
        assert!(!p.requires_grad());
        l.sum().add(&u.sum()).unwrap().backward();
        assert_eq!(a.grad().unwrap().shape(), a.shape());
    }
    let huge = double(&[], &[1usize << 60, 0, 0]).requires_grad_(true).unwrap();
    let (factor, pivots) = linalg::lu_factor(&huge).unwrap();
    factor.sum().backward();
    assert_eq!(pivots.shape(), &[1usize << 60, 0]);
    assert!(linalg::lu_factor_ex(&huge, false).is_err());
    assert!(linalg::lu(&double(&[], &[1usize << 30, 0])).is_err());
    let output = linalg::lu_solve(&factor, &pivots, &huge, true, false).unwrap();
    assert_eq!(output.shape(), huge.shape());
    assert!(ferro_core::capture(|| linalg::lu_factor(&a).map(|result| result.0)).is_err());
}

#[test]
fn lu_saved_output_metadata_and_unpacked_sibling_versions() {
    for selected in 0..4 {
        let a = double(&[0.2, 2.1, 1.1, 0.3], &[2, 2]).requires_grad_(true).unwrap();
        let (factor, pivots, info) = linalg::lu_factor_ex(&a, false).unwrap();
        [&a, &factor, &pivots, &info][selected]._bump_version_for_test();
        let result = factor.sum().grad_wrt(&[&a], false);
        assert_eq!(result.is_ok(), selected == 0 || selected == 3);
    }
    for selected in 0..3 { for differentiated in [1, 2] {
        let a = double(&[0.2, 2.1, 1.1, 0.3], &[2, 2]).requires_grad_(true).unwrap();
        let (p, l, u) = linalg::lu(&a).unwrap();
        let outputs = [p, l, u];
        outputs[selected]._bump_version_for_test();
        assert!(outputs[differentiated].sum().grad_wrt(&[&a], false).is_err());
    } }
    for selected in 0..4 {
        let a = double(&[0.2, 2.1, 1.1, 0.3], &[2, 2]);
        let (factor, pivots) = linalg::lu_factor(&a).unwrap();
        let factor = factor.requires_grad_(true).unwrap();
        let b = double(&[0.2, 0.3], &[2, 1]).requires_grad_(true).unwrap();
        let out = linalg::lu_solve(&factor, &pivots, &b, true, false).unwrap();
        [&factor, &pivots, &b, &out][selected]._bump_version_for_test();
        assert_eq!(out.sum().grad_wrt(&[&factor, &b], false).is_ok(), selected == 2);
    }
}
