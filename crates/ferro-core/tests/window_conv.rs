use ferro_core::{ops_ext::window::Window2d, Tensor};

// Independent direct spatial oracle, no production geometry or lowering.
fn oracle(
    x: &Tensor,
    w: &Tensor,
    s: [usize; 2],
    p: [usize; 2],
    d: [usize; 2],
    groups: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a = x.shape();
    let b = w.shape();
    let (n, cin, h, width) = (a[0], a[1], a[2], a[3]);
    let (cout, ic, kh, kw) = (b[0], b[1], b[2], b[3]);
    let oh = (h + 2 * p[0] - (kh - 1) * d[0] - 1) / s[0] + 1;
    let ow = (width + 2 * p[1] - (kw - 1) * d[1] - 1) / s[1] + 1;
    let xv = x.to_vec();
    let wv = w.to_vec();
    let mut y = vec![0.; n * cout * oh * ow];
    let mut dx = vec![0.; xv.len()];
    let mut dw = vec![0.; wv.len()];
    for batch in 0..n {
        for co in 0..cout {
            for row in 0..oh {
                for col in 0..ow {
                    let yi = ((batch * cout + co) * oh + row) * ow + col;
                    let g = (yi % 5) as f32 - 2.;
                    for ci in 0..ic {
                        for r in 0..kh {
                            for c in 0..kw {
                                let iy = (row * s[0] + r * d[0]) as isize - p[0] as isize;
                                let ix = (col * s[1] + c * d[1]) as isize - p[1] as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= width as isize {
                                    continue;
                                }
                                let channel = co / (cout / groups) * ic + ci;
                                let xi = ((batch * cin + channel) * h + iy as usize) * width
                                    + ix as usize;
                                let wi = ((co * ic + ci) * kh + r) * kw + c;
                                y[yi] += xv[xi] * wv[wi];
                                dx[xi] += g * wv[wi];
                                dw[wi] += g * xv[xi];
                            }
                        }
                    }
                }
            }
        }
    }
    (y, dx, dw)
}

#[test]
fn rectangular_group_depthwise_oracle_exact() {
    for (n, cin, cout, groups, s, p, d) in [
        (2, 4, 6, 2, [1, 2], [2, 1], [2, 1]),
        (2, 3, 6, 3, [2, 1], [1, 2], [1, 2]),
        (1, 2, 2, 2, [1, 1], [3, 3], [2, 2]),
        (1, 2, 3, 1, [1, 1], [0, 0], [1, 1]),
    ] {
        // Small integers make every forward and gradient sum exactly representable.
        let x = Tensor::from_vec(
            (0..n * cin * 4 * 5).map(|i| (i % 7) as f32 - 3.).collect(),
            &[n, cin, 4, 5],
        )
        .unwrap()
        .requires_grad_(true)
        .unwrap();
        let w = Tensor::from_vec(
            (0..cout * (cin / groups) * 2 * 3)
                .map(|i| (i % 5) as f32 - 2.)
                .collect(),
            &[cout, cin / groups, 2, 3],
        )
        .unwrap()
        .requires_grad_(true)
        .unwrap();
        let (ey, ex, ew) = oracle(&x, &w, s, p, d, groups);
        let y = x.conv2d_with_options(&w, s, p, d, groups).unwrap();
        assert_eq!(y.device(), ferro_core::Device::Cpu);
        assert_eq!(y.to_vec(), ey);
        let g = Tensor::from_vec(
            (0..y.numel()).map(|i| (i % 5) as f32 - 2.).collect(),
            y.shape(),
        )
        .unwrap();
        y.mul(&g).unwrap().sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), ex);
        assert_eq!(w.grad().unwrap().to_vec(), ew);
    }
}

fn weighted(y: Tensor) -> Tensor {
    let coeff = Tensor::from_vec(
        (0..y.numel())
            .map(|i| ((i % 7) as f32 - 3.) / 16.)
            .collect(),
        y.shape(),
    )
    .unwrap();
    y.mul(&coeff).unwrap().sum()
}

#[test]
fn grouped_and_depthwise_finite_differences() {
    for (cin, cout, groups) in [(4, 4, 2), (2, 4, 2)] {
        let x = Tensor::from_vec(
            (0..cin * 3 * 4)
                .map(|i| (i % 11) as f32 / 8. - 0.6)
                .collect(),
            &[1, cin, 3, 4],
        )
        .unwrap();
        let w = Tensor::from_vec(
            (0..cout * (cin / groups) * 4)
                .map(|i| (i % 7) as f32 / 8. - 0.4)
                .collect(),
            &[cout, cin / groups, 2, 2],
        )
        .unwrap();
        ferro_core::testkit::grad_check_strict(&[x, w], |t| {
            weighted(
                t[0].conv2d_with_options(&t[1], [1, 2], [1, 1], [2, 1], groups)
                    .unwrap(),
            )
        });
    }
}

#[test]
fn window_fold_and_unfold_finite_differences() {
    let x = Tensor::from_vec(
        (0..24).map(|i| (i % 9) as f32 / 8. - 0.4).collect(),
        &[1, 2, 3, 4],
    )
    .unwrap();
    ferro_core::testkit::grad_check_strict(&[x.clone()], |t| {
        weighted(t[0].unfold2d([2, 2], [1, 2], [1, 1], [2, 1]).unwrap())
    });
    let col = x.unfold2d([2, 2], [1, 2], [1, 1], [2, 1]).unwrap();
    ferro_core::testkit::grad_check_strict(&[col], |t| {
        weighted(t[0].fold2d([3, 4], [2, 2], [1, 2], [1, 1], [2, 1]).unwrap())
    });
}

#[test]
fn noncontiguous_input_weight_and_columns() {
    let x = Tensor::from_vec(
        (0..40).map(|i| (i % 7) as f32 - 3.).collect(),
        &[1, 2, 5, 4],
    )
    .unwrap()
    .transpose(2, 3)
    .unwrap()
    .requires_grad_(true)
    .unwrap();
    let w = Tensor::from_vec(
        (0..24).map(|i| (i % 5) as f32 - 2.).collect(),
        &[4, 1, 3, 2],
    )
    .unwrap()
    .transpose(2, 3)
    .unwrap()
    .requires_grad_(true)
    .unwrap();
    let (ey, ex, ew) = oracle(&x, &w, [1, 2], [1, 1], [2, 1], 2);
    let y = x
        .conv2d_with_options(&w, [1, 2], [1, 1], [2, 1], 2)
        .unwrap();
    assert_eq!(y.to_vec(), ey);
    let g = Tensor::from_vec(
        (0..y.numel()).map(|i| (i % 5) as f32 - 2.).collect(),
        y.shape(),
    )
    .unwrap();
    y.mul(&g).unwrap().sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), ex);
    assert_eq!(w.grad().unwrap().to_vec(), ew);
    let col = x.unfold2d([2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    let contig = Tensor::from_vec(x.to_vec(), x.shape()).unwrap();
    assert_eq!(
        col.to_vec(),
        contig
            .unfold2d([2, 2], [1, 1], [0, 0], [1, 1])
            .unwrap()
            .to_vec()
    );
    let cv = Tensor::from_vec((0..96).map(|i| i as f32).collect(), &[1, 12, 8])
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let cc = Tensor::from_vec(cv.to_vec(), cv.shape()).unwrap();
    assert_eq!(
        cv.fold2d([4, 5], [2, 2], [1, 1], [0, 0], [1, 1])
            .unwrap()
            .to_vec(),
        cc.fold2d([4, 5], [2, 2], [1, 1], [0, 0], [1, 1])
            .unwrap()
            .to_vec()
    );
}

#[test]
fn validation_and_empty_batch() {
    use ferro_core::Error;
    let x = Tensor::zeros(&[1, 2, 3, 4]);
    let w = Tensor::zeros(&[4, 1, 2, 2]);
    for (s, p, d, g) in [
        ([0, 1], [0, 0], [1, 1], 2),
        ([1, 0], [0, 0], [1, 1], 2),
        ([1, 1], [0, 0], [0, 1], 2),
        ([1, 1], [0, 0], [1, 0], 2),
        ([1, 1], [usize::MAX, 0], [1, 1], 2),
        ([1, 1], [0, 0], [usize::MAX, 1], 2),
        ([1, 1], [0, 0], [1, 1], 0),
        ([1, 1], [0, 0], [1, 1], 3),
        ([1, 1], [0, 0], [1, 1], 1),
    ] {
        assert!(x.conv2d_with_options(&w, s, p, d, g).is_err());
    }
    for (xs, ws) in [
        (vec![1, 2, 3], vec![4, 1, 2, 2]),
        (vec![1, 0, 3, 4], vec![4, 0, 2, 2]),
        (vec![1, 2, 0, 4], vec![4, 1, 2, 2]),
        (vec![1, 2, 3, 4], vec![0, 1, 2, 2]),
        (vec![1, 2, 3, 4], vec![4, 1, 0, 2]),
        (vec![1, 2, 3, 4], vec![4, 1, 7, 2]),
    ] {
        assert!(Tensor::zeros(&xs)
            .conv2d_with_options(&Tensor::zeros(&ws), [1, 1], [0, 0], [1, 1], 2)
            .is_err());
    }
    let int = Tensor::from_vec_i64(vec![1; 24], &[1, 2, 3, 4]).unwrap();
    assert!(matches!(
        int.conv2d_with_options(&w, [1, 1], [0, 0], [1, 1], 2),
        Err(Error::DtypeMismatch { .. })
    ));
    assert!(matches!(
        x.conv2d_with_options(
            &w.to_dtype(ferro_core::DType::I64),
            [1, 1],
            [0, 0],
            [1, 1],
            2
        ),
        Err(Error::DtypeMismatch { .. })
    ));
    assert!(matches!(
        int.unfold2d([1, 1], [1, 1], [0, 0], [1, 1]),
        Err(Error::DtypeMismatch { .. })
    ));
    assert!(int.fold2d([3, 4], [1, 1], [1, 1], [0, 0], [1, 1]).is_err());
    assert!(Tensor::zeros(&[1, 5, 6])
        .fold2d([3, 4], [2, 2], [1, 1], [0, 0], [1, 1])
        .is_err());
    assert!(Tensor::zeros(&[1, 4, 5])
        .fold2d([3, 4], [2, 2], [1, 1], [0, 0], [1, 1])
        .is_err());
    assert!(Window2d::new([usize::MAX, 2], [1, 1], [1, 1], [0, 0], [1, 1]).is_err());
    assert!(Window2d::new([1, 1], [1, 1], [1, 1], [usize::MAX / 4, 0], [1, 1]).is_err());
    let empty = Tensor::zeros(&[0, 2, 3, 4]).requires_grad_(true).unwrap();
    let w = w.requires_grad_(true).unwrap();
    let y = empty
        .conv2d_with_options(&w, [1, 1], [0, 0], [1, 1], 2)
        .unwrap();
    assert_eq!(y.shape(), &[0, 4, 2, 3]);
    y.sum().backward();
    assert!(empty.grad().unwrap().to_vec().is_empty());
    assert_eq!(w.grad().unwrap().to_vec(), vec![0.; 16]);
    let col = empty.unfold2d([2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    assert_eq!(col.shape(), &[0, 8, 6]);
    assert_eq!(
        col.fold2d([3, 4], [2, 2], [1, 1], [0, 0], [1, 1])
            .unwrap()
            .shape(),
        &[0, 2, 3, 4]
    );
}

#[test]
fn grouped_dilated_forward_backward_exact() {
    let x = Tensor::from_vec((1..=24).map(|v| v as f32).collect(), &[1, 2, 3, 4])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let w = Tensor::from_vec(vec![1., 2., -1., 3.], &[2, 1, 1, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let y = x
        .conv2d_with_options(&w, [1, 1], [0, 0], [1, 2], 2)
        .unwrap();
    assert_eq!(y.shape(), &[1, 2, 3, 2]);
    assert_eq!(
        y.to_vec(),
        vec![7., 10., 19., 22., 31., 34., 32., 34., 40., 42., 48., 50.]
    );
    y.sum().backward();
    assert_eq!(
        x.grad().unwrap().to_vec(),
        vec![
            1., 1., 2., 2., 1., 1., 2., 2., 1., 1., 2., 2., -1., -1., 3., 3., -1., -1., 3., 3.,
            -1., -1., 3., 3.
        ]
    );
    assert_eq!(w.grad().unwrap().to_vec(), vec![33., 45., 105., 117.]);
}

#[test]
fn window_overlap_values_and_adjoint() {
    let geometry = Window2d::new([3, 3], [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    assert_eq!(geometry.output_size(), [2, 2]);
    let x = Tensor::from_vec((1..=9).map(|v| v as f32).collect(), &[1, 1, 3, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let col = x.unfold2d([2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    assert_eq!(col.shape(), &[1, 4, 4]);
    assert_eq!(
        col.to_vec(),
        vec![1., 2., 4., 5., 2., 3., 5., 6., 4., 5., 7., 8., 5., 6., 8., 9.]
    );
    let folded = col.fold2d([3, 3], [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
    assert_eq!(folded.to_vec(), vec![1., 4., 3., 8., 20., 12., 7., 16., 9.]);
    col.sum().backward();
    assert_eq!(
        x.grad().unwrap().to_vec(),
        vec![1., 2., 1., 2., 4., 2., 1., 2., 1.]
    );
}
