use ferro_core::testkit::grad_check;
use ferro_core::Tensor;
use std::time::Instant;

#[test]
fn conv2d_identity() {
    // 1x1 kernel with weight 1 reproduces the input.
    let x = Tensor::from_vec(vec![1.0, -2.0, 3.0, 4.0, 0.5, -6.0], &[1, 1, 2, 3]).unwrap();
    let w = Tensor::from_vec(vec![1.0], &[1, 1, 1, 1]).unwrap();
    let y = x.conv2d(&w, 1, 0).unwrap();
    assert_eq!(y.shape(), &[1, 1, 2, 3]);
    assert_eq!(y.to_vec(), x.to_vec());

    // 3x3 input, 2x2 kernel, stride 1, padding 0, hand-computed:
    // [[1,2],[4,5]].[[1,2],[3,4]] = 37, etc.
    let x = Tensor::from_vec((1..=9).map(|v| v as f32).collect(), &[1, 1, 3, 3]).unwrap();
    let w = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
    let y = x.conv2d(&w, 1, 0).unwrap();
    assert_eq!(y.shape(), &[1, 1, 2, 2]);
    assert_eq!(y.to_vec(), vec![37.0, 47.0, 67.0, 77.0]);
}

#[test]
fn conv2d_padding_stride() {
    // 3x3 input, 2x2 ones kernel, padding 1, stride 2 -> out 2x2.
    let x = Tensor::from_vec((1..=9).map(|v| v as f32).collect(), &[1, 1, 3, 3]).unwrap();
    let w = Tensor::from_vec(vec![1.0; 4], &[1, 1, 2, 2]).unwrap();
    let y = x.conv2d(&w, 2, 1).unwrap();
    assert_eq!(y.shape(), &[1, 1, 2, 2]);
    // Top-left output tap covers only input[0][0]=1; the other three taps
    // land in the zero pad.
    assert_eq!(y.to_vec(), vec![1.0, 5.0, 11.0, 28.0]);
}

#[test]
fn conv2d_errors() {
    let x = Tensor::from_vec(vec![0.0; 4], &[1, 1, 2, 2]).unwrap();
    // rank mismatch
    let w3 = Tensor::from_vec(vec![0.0; 4], &[1, 2, 2]).unwrap();
    assert!(x.conv2d(&w3, 1, 0).is_err());
    let x3 = Tensor::from_vec(vec![0.0; 4], &[1, 2, 2]).unwrap();
    let w = Tensor::from_vec(vec![0.0; 4], &[1, 1, 2, 2]).unwrap();
    assert!(x3.conv2d(&w, 1, 0).is_err());
    // channel mismatch: c_in 1 vs weight expecting 2
    let w2c = Tensor::from_vec(vec![0.0; 8], &[1, 2, 2, 2]).unwrap();
    assert!(x.conv2d(&w2c, 1, 0).is_err());
    // kernel larger than (padded) input
    let wbig = Tensor::from_vec(vec![0.0; 16], &[1, 1, 4, 4]).unwrap();
    assert!(x.conv2d(&wbig, 1, 0).is_err());
    let wbig5 = Tensor::from_vec(vec![0.0; 25], &[1, 1, 5, 5]).unwrap();
    assert!(x.conv2d(&wbig5, 1, 1).is_err());
    // zero stride
    assert!(x.conv2d(&w, 0, 0).is_err());
}

// Weighted-sum loss so every output element gets a distinct gradient.
fn weighted_loss(y: Tensor) -> Tensor {
    let numel: usize = y.shape().iter().product();
    let coeffs: Vec<f32> = (0..numel).map(|i| 0.1 + 0.3 * i as f32).collect();
    let c = Tensor::from_vec(coeffs, y.shape()).unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn conv2d_grad() {
    // stride 1, padding 0
    let xv: Vec<f32> = (0..16).map(|i| 0.5 - 0.13 * i as f32).collect();
    let x = Tensor::from_vec(xv, &[1, 1, 4, 4]).unwrap();
    let w = Tensor::from_vec(vec![0.7, -0.4, 0.2, 1.1], &[1, 1, 2, 2]).unwrap();
    grad_check(&[x, w], |t| weighted_loss(t[0].conv2d(&t[1], 1, 0).unwrap()));

    // padding 1, stride 2
    let xv: Vec<f32> = (0..9).map(|i| -0.6 + 0.21 * i as f32).collect();
    let x = Tensor::from_vec(xv, &[1, 1, 3, 3]).unwrap();
    let w = Tensor::from_vec(vec![0.9, -0.3, 0.5, -1.2], &[1, 1, 2, 2]).unwrap();
    grad_check(&[x, w], |t| weighted_loss(t[0].conv2d(&t[1], 2, 1).unwrap()));

    // multichannel: 2 in channels, 2 out channels
    let xv: Vec<f32> = (0..18).map(|i| 0.3 - 0.11 * i as f32).collect();
    let wv: Vec<f32> = (0..16).map(|i| -0.5 + 0.17 * i as f32).collect();
    let x = Tensor::from_vec(xv, &[1, 2, 3, 3]).unwrap();
    let w = Tensor::from_vec(wv, &[2, 2, 2, 2]).unwrap();
    grad_check(&[x, w], |t| weighted_loss(t[0].conv2d(&t[1], 1, 0).unwrap()));
}

#[test]
fn conv2d_grad_odd_configs() {
    // stride 2, padding 2.
    let xv: Vec<f32> = (0..2 * 5 * 4).map(|i| 0.4 - 0.031 * i as f32).collect();
    let wv: Vec<f32> = (0..2 * 2 * 3 * 3).map(|i| -0.35 + 0.02 * i as f32).collect();
    let x = Tensor::from_vec(xv, &[1, 2, 5, 4]).unwrap();
    let w = Tensor::from_vec(wv, &[2, 2, 3, 3]).unwrap();
    grad_check(&[x, w], |t| weighted_loss(t[0].conv2d(&t[1], 2, 2).unwrap()));

    // 1x1 kernel.
    let xv: Vec<f32> = (0..2 * 3 * 3 * 3).map(|i| -0.4 + 0.017 * i as f32).collect();
    let wv: Vec<f32> = (0..2 * 3).map(|i| 0.5 - 0.15 * i as f32).collect();
    let x = Tensor::from_vec(xv, &[2, 3, 3, 3]).unwrap();
    let w = Tensor::from_vec(wv, &[2, 3, 1, 1]).unwrap();
    grad_check(&[x, w], |t| weighted_loss(t[0].conv2d(&t[1], 1, 0).unwrap()));
}

fn seq(n: usize, offset: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| offset + scale * i as f32).collect()
}

fn assert_close(a: f32, b: f32, tol: f32, what: &str) {
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!((a - b).abs() <= tol * scale, "{what}: {a} vs {b} (tol {tol}, scale {scale})");
}

// Pre-im2col naive direct conv, kept only as a reference oracle for the
// parity test below (mirrors the loop nest conv2d used before the rewrite).
fn naive_conv2d(x: &Tensor, weight: &Tensor, stride: usize, padding: usize) -> Tensor {
    let (in_shape, w_shape) = (x.shape(), weight.shape());
    let (n, c_in, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let (c_out, kh, kw) = (w_shape[0], w_shape[2], w_shape[3]);
    let (ph, pw) = (h + 2 * padding, w + 2 * padding);
    let out_h = (ph - kh) / stride + 1;
    let out_w = (pw - kw) / stride + 1;

    let xv = x.to_vec();
    let wv = weight.to_vec();
    let mut out = vec![0.0f32; n * c_out * out_h * out_w];
    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut s = 0.0f32;
                    for ci in 0..c_in {
                        for r in 0..kh {
                            let ih = (oh * stride + r).wrapping_sub(padding);
                            if ih >= h {
                                continue;
                            }
                            for c in 0..kw {
                                let iw = (ow * stride + c).wrapping_sub(padding);
                                if iw >= w {
                                    continue;
                                }
                                let xi = ((ni * c_in + ci) * h + ih) * w + iw;
                                let wi = ((co * c_in + ci) * kh + r) * kw + c;
                                s += xv[xi] * wv[wi];
                            }
                        }
                    }
                    out[((ni * c_out + co) * out_h + oh) * out_w + ow] = s;
                }
            }
        }
    }
    Tensor::from_vec(out, &[n, c_out, out_h, out_w]).unwrap()
}

struct ConvCase {
    n: usize,
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
}

#[test]
fn conv2d_matches_naive_forward() {
    let cases = [
        // 1x1 kernel.
        ConvCase { n: 1, cin: 3, cout: 8, h: 5, w: 7, kh: 1, kw: 1, stride: 1, padding: 0 },
        ConvCase { n: 2, cin: 3, cout: 1, h: 7, w: 9, kh: 1, kw: 1, stride: 2, padding: 0 },
        // kernel == padded input size.
        ConvCase { n: 2, cin: 3, cout: 3, h: 5, w: 5, kh: 5, kw: 5, stride: 1, padding: 0 },
        ConvCase { n: 1, cin: 1, cout: 3, h: 3, w: 4, kh: 5, kw: 6, stride: 1, padding: 1 },
        // padding >= kernel - 1.
        ConvCase { n: 1, cin: 1, cout: 3, h: 4, w: 6, kh: 3, kw: 3, stride: 1, padding: 3 },
        ConvCase { n: 1, cin: 3, cout: 3, h: 3, w: 3, kh: 2, kw: 2, stride: 1, padding: 5 },
        // stride grid, non-square h != w, varied channel counts.
        ConvCase { n: 2, cin: 8, cout: 3, h: 9, w: 6, kh: 3, kw: 3, stride: 2, padding: 1 },
        ConvCase { n: 1, cin: 3, cout: 1, h: 10, w: 13, kh: 4, kw: 4, stride: 3, padding: 2 },
        ConvCase { n: 1, cin: 1, cout: 1, h: 6, w: 5, kh: 2, kw: 3, stride: 1, padding: 1 },
        ConvCase { n: 2, cin: 8, cout: 8, h: 8, w: 11, kh: 3, kw: 3, stride: 1, padding: 1 },
    ];

    for (i, case) in cases.iter().enumerate() {
        let xv = seq(case.n * case.cin * case.h * case.w, -1.3, 0.037 + i as f32 * 0.001);
        let wv = seq(case.cout * case.cin * case.kh * case.kw, 0.7, -0.021 - i as f32 * 0.0007);
        let x = Tensor::from_vec(xv, &[case.n, case.cin, case.h, case.w]).unwrap();
        let w = Tensor::from_vec(wv, &[case.cout, case.cin, case.kh, case.kw]).unwrap();

        let expect = naive_conv2d(&x, &w, case.stride, case.padding);
        let got = x.conv2d(&w, case.stride, case.padding).unwrap();
        assert_eq!(got.shape(), expect.shape(), "case {i} shape");
        for (j, (gv, ev)) in got.to_vec().iter().zip(expect.to_vec().iter()).enumerate() {
            assert_close(*gv, *ev, 1e-4, &format!("case {i} elem {j}"));
        }
    }
}

// Ignored by default (slow, release-only); run manually with:
// cargo test -p ferro-core --release -- --ignored conv_timing --nocapture
#[test]
#[ignore]
fn conv_timing_im2col_vs_naive() {
    let (n, cin, cout, h, w, kh, kw, stride, padding) = (4, 32, 64, 56, 56, 3, 3, 1, 1);
    let xv = seq(n * cin * h * w, -0.7, 0.0031);
    let wv = seq(cout * cin * kh * kw, 0.4, -0.0021);
    let x = Tensor::from_vec(xv, &[n, cin, h, w]).unwrap();
    let wt = Tensor::from_vec(wv, &[cout, cin, kh, kw]).unwrap();

    let t0 = Instant::now();
    let naive_out = naive_conv2d(&x, &wt, stride, padding);
    let naive_dt = t0.elapsed();

    let t1 = Instant::now();
    let new_out = x.conv2d(&wt, stride, padding).unwrap();
    let new_dt = t1.elapsed();

    for (a, b) in naive_out.to_vec().iter().zip(new_out.to_vec().iter()) {
        assert_close(*a, *b, 1e-4, "timing parity");
    }

    println!("naive: {naive_dt:?}, im2col+gemm: {new_dt:?}, speedup: {:.1}x",
        naive_dt.as_secs_f64() / new_dt.as_secs_f64());
    assert!(new_dt < naive_dt, "im2col+gemm should be faster than naive direct conv");
}
