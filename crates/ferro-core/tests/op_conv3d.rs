use ferro_core::{testkit::grad_check, Tensor};

fn r(n: usize, shape: &[usize], seed: usize) -> Tensor {
    Tensor::from_vec((0..n).map(|i| (((i * 7 + seed * 13) % 19) as f32 - 9.0) * 0.1).collect(), shape).unwrap()
}

/// Depth-slice composition of native conv2d: the semantics the op replaces.
fn reference(x: &Tensor, w: &Tensor, s: [usize; 3], p: [usize; 3], d: [usize; 3], groups: usize) -> Vec<f32> {
    let (xs, ws) = (x.shape().to_vec(), w.shape().to_vec());
    let od = (xs[2] + 2 * p[0] - d[0] * (ws[2] - 1) - 1) / s[0] + 1;
    let mut planes = Vec::new();
    for o in 0..od {
        let mut plane: Option<Tensor> = None;
        for k in 0..ws[2] {
            let src = (o * s[0] + k * d[0]) as isize - p[0] as isize;
            if src < 0 || src as usize >= xs[2] { continue; }
            let term = x.narrow(2, src as usize, 1).unwrap().squeeze(2).unwrap()
                .conv2d_with_options(&w.narrow(2, k, 1).unwrap().squeeze(2).unwrap(), [s[1], s[2]], [p[1], p[2]], [d[1], d[2]], groups).unwrap();
            plane = Some(match plane { Some(acc) => acc.add(&term).unwrap(), None => term });
        }
        planes.push(plane.unwrap().unsqueeze(2).unwrap());
    }
    Tensor::cat(&planes, 2).unwrap().to_vec()
}

#[test]
fn matches_depth_slice_composition_across_options() {
    for (s, p, d, groups) in [([1; 3], [0; 3], [1; 3], 1), ([2, 1, 2], [1, 0, 1], [1, 2, 1], 2), ([1, 2, 1], [2, 1, 0], [2, 1, 1], 1)] {
        let x = r(2 * 4 * 5 * 6 * 5, &[2, 4, 5, 6, 5], 1);
        let w = r(6 * (4 / groups) * 2 * 3 * 2, &[6, 4 / groups, 2, 3, 2], 2);
        let y = x.conv3d_with_options(&w, s, p, d, groups).unwrap();
        let expected = reference(&x, &w, s, p, d, groups);
        assert_eq!(y.numel(), expected.len());
        for (a, b) in y.to_vec().iter().zip(&expected) { assert!((a - b).abs() < 1e-4, "{a} vs {b}"); }
    }
}

#[test]
fn gradients_match_finite_differences() {
    let x = r(1 * 2 * 3 * 4 * 3, &[1, 2, 3, 4, 3], 3);
    let w = r(2 * 1 * 2 * 2 * 2, &[2, 1, 2, 2, 2], 4);
    let probe = r(2 * 3 * 3 * 2, &[1, 2, 3, 3, 2], 5);
    grad_check(&[x, w], |a| a[0].conv3d_with_options(&a[1], [1, 2, 1], [1, 1, 0], [2, 1, 1], 2).unwrap().mul(&probe).unwrap().sum());
}

#[test]
fn rejects_bad_geometry_and_dtypes() {
    let x = Tensor::zeros(&[1, 2, 3, 3, 3]);
    assert!(x.conv3d_with_options(&Tensor::zeros(&[2, 2, 4, 1, 1]), [1; 3], [0; 3], [1; 3], 1).is_err());
    assert!(x.conv3d_with_options(&Tensor::zeros(&[3, 1, 1, 1, 1]), [1; 3], [0; 3], [1; 3], 2).is_err());
    assert!(x.conv3d_with_options(&Tensor::zeros(&[2, 2, 1, 1, 1]), [0, 1, 1], [0; 3], [1; 3], 1).is_err());
    assert!(x.conv3d_with_options(&Tensor::zeros(&[2, 2, 1, 1]), [1; 3], [0; 3], [1; 3], 1).is_err());
    assert!(Tensor::from_vec_i64(vec![0; 27], &[1, 1, 3, 3, 3]).unwrap().conv3d_with_options(&Tensor::zeros(&[1, 1, 1, 1, 1]), [1; 3], [0; 3], [1; 3], 1).is_err());
}
