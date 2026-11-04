use ferro_core::{DType, Tensor};

fn equivalent(actual: &[f64], expected: &[f64]) {
    for (a, b) in actual.iter().zip(expected) { assert!(a == b || a.is_nan() && b.is_nan(), "{a} != {b}"); }
}

#[test]
fn relu_and_abs_match_nonfinite_forward_inputs_and_cotangents() {
    for dtype in [DType::F32, DType::F64] { for seed in [1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let x = Tensor::from_vec_f64(vec![f64::NAN, -0.0, 0.0, -0.2, 0.2], &[5]).unwrap().to_dtype(dtype).requires_grad_(true).unwrap();
        let g = Tensor::from_vec_f64(vec![seed; 5], &[5]).unwrap().to_dtype(dtype);
        let relu = x.relu().vjp_wrt(&[&x], &g, false).unwrap().remove(0);
        equivalent(&relu.to_vec_f64(), &[seed, 0., 0., 0., seed]);
        let abs = x.abs().vjp_wrt(&[&x], &g, false).unwrap().remove(0);
        equivalent(&abs.to_vec_f64(), &[seed * 0., seed * 0., seed * 0., -seed, seed]);
        assert_eq!(x.sign().unwrap().to_vec_f64(), vec![0., 0., 0., -1., 1.]);
        assert_eq!(x.sign().unwrap().vjp_wrt(&[&x], &g, false).unwrap()[0].to_vec_f64(), vec![0.; 5]);
    } }
}

#[test]
fn piecewise_graph_derivatives_retain_real_input_and_seed_dependencies() {
    let x = Tensor::from_vec(vec![-0.4, 0., 0.7, f32::NAN], &[4]).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::from_vec(vec![0.2, 0.3, 0.4, 0.5], &[4]).unwrap().requires_grad_(true).unwrap();
    for (output, expected_seed) in [(x.relu(), vec![0., 0., 1., 1.]), (x.abs(), vec![-1., 0., 1., 0.])] {
        let gradient = output.vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
        assert!(gradient.requires_grad() && gradient.depends_on(&x) && gradient.depends_on(&seed));
        let second = gradient.sum().grad_wrt(&[&x, &seed], false).unwrap();
        assert_eq!(second[0].to_vec(), vec![0.; 4]);
        assert_eq!(second[1].to_vec(), expected_seed);
        let optional = gradient.sum().vjp_wrt_optional(&[&x], &Tensor::scalar(1.), false).unwrap();
        assert!(optional[0].is_some(), "a true zero derivative is not an unused input");
    }
}

#[test]
fn relu_graph_derivative_suppresses_nonfinite_seed_where_inactive() {
    let x = Tensor::from_vec(vec![-1., 0., 1., f32::NAN], &[4]).unwrap().requires_grad_(true).unwrap();
    for value in [f32::NAN, f32::INFINITY] {
        let seed = Tensor::full(&[4], value).requires_grad_(true).unwrap();
        let g = x.relu().vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
        equivalent(&g.to_vec_f64(), &[0., 0., value as f64, value as f64]);
    }
}

#[test]
fn relu_zero_higher_derivatives_keep_the_saved_result_dependency() {
    let x = Tensor::from_vec(vec![-0.4, 0.0, 0.7, f32::NAN], &[4]).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::ones(&[4]).requires_grad_(true).unwrap();
    let mut gradient = x.relu().vjp_wrt(&[&x], &seed, true).unwrap().remove(0);
    for _order in 2..=5 {
        gradient = gradient.sum().grad_wrt(&[&x], true).unwrap().remove(0);
        assert_eq!(gradient.to_vec(), vec![0.; 4]);
        assert!(gradient.requires_grad() && gradient.depends_on(&x));
        assert!(!gradient.depends_on(&seed));
    }
    for output in [x.abs(), x.sign().unwrap()] {
        let gradient = output.sum().grad_wrt(&[&x], true).unwrap().remove(0);
        let zero = if gradient.requires_grad() { gradient.sum().grad_wrt(&[&x], true).unwrap().remove(0) } else { gradient };
        assert_eq!(zero.to_vec(), vec![0.; 4]);
        assert!(!zero.requires_grad());
    }
}
