use ferro_core::philox::Philox;
use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn dropout_same_seed_offset_is_bitwise_identical() {
    let x = Tensor::from_vec(vec![1.0; 256], &[256]).unwrap();
    let a = x.dropout(0.5, true, 7, 3).unwrap().to_vec();
    let b = x.dropout(0.5, true, 7, 3).unwrap().to_vec();
    assert_eq!(a, b);
}

#[test]
fn dropout_different_offset_differs() {
    let x = Tensor::from_vec(vec![1.0; 256], &[256]).unwrap();
    let a = x.dropout(0.5, true, 7, 3).unwrap().to_vec();
    let b = x.dropout(0.5, true, 7, 4).unwrap().to_vec();
    assert_ne!(a, b);
}

#[test]
fn dropout_eval_mode_is_identity() {
    let x = Tensor::from_vec(vec![1.0, -2.0, 3.5, 0.0], &[4]).unwrap();
    let got = x.dropout(0.5, false, 7, 0).unwrap();
    assert_eq!(got.to_vec(), x.to_vec());

    let got_p0 = x.dropout(0.0, true, 7, 0).unwrap();
    assert_eq!(got_p0.to_vec(), x.to_vec());
}

#[test]
fn dropout_rejects_invalid_p() {
    let x = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    assert!(x.dropout(1.0, true, 0, 0).is_err());
    assert!(x.dropout(-0.1, true, 0, 0).is_err());
}

#[test]
fn dropout_scaling_and_zero_fraction() {
    let n = 100_000;
    let x = Tensor::from_vec(vec![2.0; n], &[n]).unwrap();
    let p = 0.3;
    let out = x.dropout(p, true, 123, 0).unwrap().to_vec();

    let mean_out: f32 = out.iter().sum::<f32>() / n as f32;
    let mean_x: f32 = 2.0;
    assert!((mean_out - mean_x).abs() < 0.05, "mean_out {mean_out}");

    let zero_frac = out.iter().filter(|&&v| v == 0.0).count() as f32 / n as f32;
    assert!((zero_frac - p).abs() < 0.02, "zero_frac {zero_frac}");
}

#[test]
fn dropout_grad_check() {
    let a = Tensor::from_vec(vec![0.5, -1.5, 2.0, 3.0, -0.25, 1.25], &[6]).unwrap();
    let w = Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5, -2.0, 1.5], &[6]).unwrap();
    grad_check(&[a], |t| {
        t[0].dropout(0.4, true, 99, 11)
            .unwrap()
            .mul(&w)
            .unwrap()
            .sum()
    });
}

#[test]
fn dropout_grad_equals_mask_bitwise() {
    let n = 32u64;
    let x = Tensor::from_vec(vec![1.0; n as usize], &[n as usize])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let seed = 55;
    let offset = 6;
    let p = 0.25;
    let out = x.dropout(p, true, seed, offset).unwrap();
    out.sum().backward();
    let grad = x.grad().unwrap().to_vec();

    let philox = Philox::new(seed);
    let scale = 1.0 / (1.0 - p);
    for i in 0..n {
        let expected = if philox.uniform_at(offset, i) < p {
            0.0
        } else {
            scale
        };
        assert_eq!(grad[i as usize], expected, "mismatch at {i}");
    }
}
