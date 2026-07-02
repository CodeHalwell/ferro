use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

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
