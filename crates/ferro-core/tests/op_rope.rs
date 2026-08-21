use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn rope_position_zero_is_identity() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let pos = Tensor::from_vec_i64(vec![0], &[1]).unwrap();
    assert_eq!(x.rope(&pos, 10000.0).unwrap().to_vec(), x.to_vec());
}

#[test]
fn rope_values_manual_rotation() {
    // head_dim=2, position 1, base 10000: the single pair rotates by 1 radian.
    let x = Tensor::from_vec(vec![1.0, 2.0, 1.0, 2.0], &[2, 2]).unwrap();
    let pos = Tensor::from_vec_i64(vec![0, 1], &[2]).unwrap();
    let got = x.rope(&pos, 10000.0).unwrap().to_vec();
    let (c, s) = (1.0f32.cos(), 1.0f32.sin());
    let want = [1.0, 2.0, c - 2.0 * s, 2.0 * c + s];
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
    }
}

#[test]
fn rope_preserves_pair_norms() {
    let x = Tensor::from_vec(vec![0.3, -1.2, 0.8, 0.5, 1.1, -0.7, 0.2, 0.9], &[2, 4]).unwrap();
    let pos = Tensor::from_vec_i64(vec![3, 17], &[2]).unwrap();
    let y = x.rope(&pos, 10000.0).unwrap().to_vec();
    let xv = x.to_vec();
    for row in 0..2 {
        for j in 0..2 {
            let (a, b) = (row * 4 + j, row * 4 + 2 + j);
            let nx = xv[a] * xv[a] + xv[b] * xv[b];
            let ny = y[a] * y[a] + y[b] * y[b];
            assert!((nx - ny).abs() < 1e-5);
        }
    }
}

#[test]
fn rope_explicit_positions_offset_decode() {
    // A single decode step at position 2 must match row 2 of a full pass.
    let full = Tensor::from_vec(vec![0.4, 1.1, -0.6, 0.9, 0.2, -1.3, 0.7, 0.5, 1.0, -0.2, 0.3, 0.8], &[3, 4]).unwrap();
    let all_pos = Tensor::from_vec_i64(vec![0, 1, 2], &[3]).unwrap();
    let want = &full.rope(&all_pos, 10000.0).unwrap().to_vec()[8..12];
    let step = Tensor::from_vec(vec![1.0, -0.2, 0.3, 0.8], &[1, 4]).unwrap();
    let step_pos = Tensor::from_vec_i64(vec![2], &[1]).unwrap();
    let got = step.rope(&step_pos, 10000.0).unwrap().to_vec();
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-6);
    }
}

#[test]
fn rope_rejects_bad_inputs() {
    let odd = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let pos = Tensor::from_vec_i64(vec![0], &[1]).unwrap();
    assert!(odd.rope(&pos, 10000.0).is_err());
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let short = Tensor::from_vec_i64(vec![0], &[1]).unwrap();
    assert!(x.rope(&short, 10000.0).is_err());
    let float_pos = Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    assert!(x.rope(&float_pos, 10000.0).is_err());
}

#[test]
fn rope_grad() {
    let x = Tensor::from_vec((0..24).map(|i| ((i * 7 % 11) as f32 - 5.0) / 4.0).collect(), &[2, 3, 4]).unwrap();
    let pos = Tensor::from_vec_i64(vec![0, 1, 2], &[3]).unwrap();
    grad_check(&[x.clone()], |t| t[0].rope(&pos, 10000.0).unwrap().mul(&t[0]).unwrap().sum());
}
