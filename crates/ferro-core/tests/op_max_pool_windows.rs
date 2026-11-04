use ferro_core::Tensor;

fn ranges(length: usize, kernel: usize, stride: usize) -> Vec<(usize, usize, usize)> {
    (0..=(length - kernel) / stride).map(|o| (o * stride, o * stride + kernel, 1)).collect()
}

#[test]
fn matches_native_max_pool2d_values_and_gradients() {
    let data: Vec<f32> = (0..2 * 3 * 6 * 5).map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.25).collect();
    let x = Tensor::from_vec(data, &[2, 3, 6, 5]).unwrap().requires_grad_(true).unwrap();
    let (flat, local) = x.max_pool_windows(&[ranges(6, 2, 2), ranges(5, 2, 2)]).unwrap();
    assert_eq!(flat.shape(), &[2, 3, 3, 2]);
    let n = flat.numel();
    let y = x.reshape(&[x.numel()]).unwrap().gather(0, &flat.reshape(&[n]).unwrap()).unwrap().reshape(&[2, 3, 3, 2]).unwrap();
    let reference = x.max_pool2d(2, 2).unwrap();
    assert_eq!(y.to_vec(), reference.to_vec());
    let plane = 30;
    for (f, l) in flat.to_vec_i64().iter().zip(local.to_vec_i64()) { assert_eq!(f % plane, l); }
    y.sum().backward();
    let g = x.grad().unwrap().to_vec();
    x.zero_grad();
    reference.sum().backward();
    assert_eq!(g, x.grad().unwrap().to_vec());
}

#[test]
fn padding_dilation_and_adaptive_ranges_select_row_major_first_max_or_nan() {
    let x = Tensor::from_vec(vec![1., 3., 3., 0., f32::NAN, 2., f32::NAN, 5., -1.], &[1, 1, 3, 3]).unwrap();
    let (_, local) = x.max_pool_windows(&[vec![(0, 1, 1), (0, 3, 2)], vec![(0, 3, 1)]]).unwrap();
    assert_eq!(local.to_vec_i64(), vec![1, 6]);
    let (_, local) = x.max_pool_windows(&[vec![(0, 3, 2)], vec![(0, 2, 1), (1, 3, 1)]]).unwrap();
    assert_eq!(local.to_vec_i64(), vec![6, 7]);
}

#[test]
fn rejects_empty_or_out_of_bounds_windows_and_non_f32() {
    let x = Tensor::zeros(&[1, 1, 4]);
    assert!(x.max_pool_windows(&[vec![(2, 2, 1)]]).is_err());
    assert!(x.max_pool_windows(&[vec![(3, 5, 1)]]).is_err());
    assert!(x.max_pool_windows(&[vec![(0, 2, 0)]]).is_err());
    assert!(x.max_pool_windows(&[vec![(0, 2, 1)], vec![(0, 1, 1)]]).is_err());
    assert!(Tensor::from_vec_i64(vec![1, 2], &[1, 1, 2]).unwrap().max_pool_windows(&[vec![(0, 2, 1)]]).is_err());
}
