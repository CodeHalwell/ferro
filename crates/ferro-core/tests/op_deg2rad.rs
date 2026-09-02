use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn deg2rad_values() {
    let a = Tensor::from_vec(vec![0.0, 90.0, 180.0], &[3]).unwrap();
    let got = a.deg2rad().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    assert!((got[2] - std::f32::consts::PI).abs() < 1e-5);
}

#[test]
fn rad2deg_values() {
    let a = Tensor::from_vec(vec![0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI], &[3]).unwrap();
    let got = a.rad2deg().unwrap().to_vec();
    assert!((got[0] - 0.0).abs() < 1e-5);
    assert!((got[1] - 90.0).abs() < 1e-5);
    assert!((got[2] - 180.0).abs() < 1e-5);
}

#[test]
fn deg2rad_rad2deg_roundtrip() {
    let a = Tensor::from_vec(vec![12.5, -47.0, 200.0, 333.25], &[2, 2]).unwrap();
    let got = a.deg2rad().unwrap().rad2deg().unwrap().to_vec();
    let want = a.to_vec();
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-3);
    }
}

#[test]
fn deg2rad_grad() {
    let a = Tensor::from_vec(vec![12.5, -47.0, 200.0, 333.25], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].deg2rad().unwrap().sum());
}

#[test]
fn rad2deg_grad() {
    let a = Tensor::from_vec(vec![0.7, -1.3, 2.1, 0.05], &[2, 2]).unwrap();
    grad_check(&[a], |t| t[0].rad2deg().unwrap().sum());
}
