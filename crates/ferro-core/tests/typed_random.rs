use ferro_core::{DType, Rng, Tensor};

#[test]
fn double_uniform_keeps_53_bits_and_preserves_existing_float_stream() {
    let single = Rng::new(17);
    let double = Rng::new(17);
    let values = Tensor::rand_f64(&[1024], &double).unwrap().to_vec_f64();
    assert!(values.iter().all(|&x| (0.0..1.0).contains(&x)));
    assert!(values.iter().any(|&x| x != (x as f32) as f64));
    for &x in &values {
        assert_eq!((x * (1u64 << 24) as f64).floor() / (1u64 << 24) as f64, single.uniform() as f64);
    }
    assert_eq!(single.state(), double.state(), "one uniform sample consumes one PRNG transition for either dtype");
}

#[test]
fn double_normals_are_finite_reproducible_and_not_narrowed() {
    let rng = Rng::new(831);
    let state = rng.state();
    let values = Tensor::randn_f64(&[8192], &rng).unwrap();
    assert_eq!(values.dtype(), DType::F64);
    let first = values.to_vec_f64();
    assert!(first.iter().all(|x| x.is_finite()));
    assert!(first.iter().any(|&x| x != (x as f32) as f64));
    let mean = first.iter().sum::<f64>() / first.len() as f64;
    let variance = first.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / first.len() as f64;
    assert!(mean.abs() < 0.05, "normal mean {mean}");
    assert!((variance - 1.0).abs() < 0.06, "normal variance {variance}");
    rng.restore_state(state).unwrap();
    assert_eq!(Tensor::randn_f64(&[8192], &rng).unwrap().to_vec_f64(), first);
}

#[test]
fn typed_random_validates_before_advancing_state_and_empty_draws_are_noops() {
    let rng = Rng::new(942);
    let before = rng.state();
    for normal in [false, true] {
        let draw = if normal { Tensor::randn_f64 } else { Tensor::rand_f64 };
        assert!(draw(&[usize::MAX], &rng).is_err());
        assert_eq!(rng.state(), before);
        let empty = draw(&[3, 0, 2], &rng).unwrap();
        assert_eq!(empty.shape(), &[3, 0, 2]);
        assert_eq!(empty.dtype(), DType::F64);
        assert_eq!(rng.state(), before);
        let x = draw(&[], &rng).unwrap().requires_grad_(true).unwrap();
        x.sum().backward();
        assert_eq!(x.grad().unwrap().to_vec_f64(), vec![1.0]);
        assert_ne!(rng.state(), before);
        rng.restore_state(before).unwrap();
    }
}
