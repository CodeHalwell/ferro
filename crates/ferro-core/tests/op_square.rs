use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn square_values() {
    let a = Tensor::from_vec(vec![2.0, -3.0, 0.5], &[3]).unwrap();
    let got = a.square().unwrap().to_vec();
    assert!((got[0] - 4.0).abs() < 1e-5);
    assert!((got[1] - 9.0).abs() < 1e-5);
    assert!((got[2] - 0.25).abs() < 1e-5);
}

#[test]
fn square_grad() {
    let a = Tensor::from_vec(vec![0.3, -0.7, 1.5, -2.0], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].square().unwrap().sum());
}

#[test]
fn square_higher_derivatives_and_direct_jvp_use_the_same_native_graph() {
    let a = Tensor::from_vec(vec![0.2, -0.7], &[2]).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::from_vec(vec![0.3, 0.4], &[2]).unwrap().requires_grad_(true).unwrap();
    let first = a.square().unwrap().vjp_wrt(&[&a], &seed, true).unwrap().remove(0);
    let second = first.sum().grad_wrt(&[&a, &seed], true).unwrap();
    assert_eq!(second[0].to_vec(), vec![0.6, 0.8]);
    assert_eq!(second[1].to_vec(), vec![0.4, -1.4]);
    assert_eq!(second[0].sum().grad_wrt(&[&seed], false).unwrap()[0].to_vec(), vec![2.; 2]);
    let level = ferro_core::forward_ad::DualLevel::new(&[&a], &[&seed]).unwrap();
    let result = a.square().unwrap();
    assert_eq!(level.tangent(&result).unwrap().unwrap().to_vec(), vec![0.12000000476837158, -0.56]);
    drop(level);
    let alpha = ferro_core::ops_ext::add_alpha::Alpha::F64(0.3);
    let loss = a.add_alpha(&seed, alpha).unwrap().tanh().square().unwrap().sum();
    let derivative = loss.grad_wrt(&[&a, &seed], true).unwrap();
    assert!(derivative[0].sum().grad_wrt(&[&a, &seed], false).is_ok());
}

#[test]
fn square_adjoint_rounds_twice_input_before_seed_multiplication() {
    let x = Tensor::from_vec(vec![f32::MAX, f32::MAX, -0.], &[3]).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::from_vec(vec![0., 0.25, 1.], &[3]).unwrap();
    for graph in [false, true] {
        let gradient = x.square().unwrap().vjp_wrt(&[&x], &seed, graph).unwrap().remove(0).to_vec();
        assert!(gradient[0].is_nan()); assert_eq!(gradient[1], f32::INFINITY);
        assert_eq!(gradient[2].to_bits(), (-0.0f32).to_bits());
    }
    let level = ferro_core::forward_ad::DualLevel::new(&[&x], &[&seed]).unwrap();
    let output = x.square().unwrap(); let gradient = level.tangent(&output).unwrap().unwrap().to_vec();
    assert!(gradient[0].is_nan()); assert_eq!(gradient[1], f32::INFINITY); drop(level);
    let x = Tensor::from_vec_f64(vec![f64::MAX, f64::MAX, -0.], &[3]).unwrap().requires_grad_(true).unwrap();
    let seed = Tensor::from_vec_f64(vec![0., 0.25, 1.], &[3]).unwrap();
    let gradient = x.square().unwrap().vjp_wrt(&[&x], &seed, false).unwrap().remove(0).to_vec_f64();
    assert!(gradient[0].is_nan()); assert_eq!(gradient[1], f64::INFINITY);
    assert_eq!(gradient[2].to_bits(), (-0.0f64).to_bits());
}

#[test]
fn square_saves_input_values_but_not_its_mutable_output() {
    let x = Tensor::from_vec(vec![0.2, 0.4], &[2]).unwrap().requires_grad_(true).unwrap();
    let y = x.square().unwrap();
    y.detach().copy_from(&Tensor::zeros(&[2])).unwrap();
    assert_eq!(y.sum().grad_wrt(&[&x], true).unwrap()[0].to_vec(), vec![0.4, 0.8]);
    x.detach().copy_from(&Tensor::ones(&[2])).unwrap();
    assert!(y.sum().grad_wrt(&[&x], false).is_err());
    assert!(x.grad().is_none());
}
