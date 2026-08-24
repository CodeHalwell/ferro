use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.14 + 0.33 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn scatter_values_dim1() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let idx = Tensor::from_vec_i64(vec![0, 1, 1, 0], &[2, 2]).unwrap();
    let src = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();

    // out[i][index[i][j]][j] = src[i][j], untouched cells keep self:
    // row 0: out[0][0]=10, out[0][1]=20 -> [10, 20]
    // row 1: out[1][1]=30, out[1][0]=40 -> [40, 30]
    let y = x.scatter(1, &idx, &src).unwrap();
    assert_eq!(y.shape(), &[2, 2]);
    assert_eq!(y.to_vec(), vec![10.0, 20.0, 40.0, 30.0]);
}

#[test]
fn scatter_values_dim0_keeps_unwritten_cells() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]).unwrap();
    let idx = Tensor::from_vec_i64(vec![0, 2, 1, 0], &[2, 2]).unwrap();
    let src = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();

    // (i,j)=(0,0): out[0][0]=10; (0,1): out[2][1]=20; (1,0): out[1][0]=30;
    // (1,1): out[0][1]=40; everything else stays self.
    let y = x.scatter(0, &idx, &src).unwrap();
    assert_eq!(
        y.to_vec(),
        vec![10.0, 40.0, 3.0, 30.0, 5.0, 6.0, 7.0, 20.0, 9.0]
    );
}

#[test]
fn scatter_duplicates_last_writer_wins() {
    // Both index rows point at out[0][0]; ascending k order means src[1] wins.
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let idx = Tensor::from_vec_i64(vec![0, 0], &[2, 1]).unwrap();
    let src = Tensor::from_vec(vec![5.0, 7.0], &[2, 1]).unwrap();
    let y = base.scatter(0, &idx, &src).unwrap();
    assert_eq!(y.to_vec(), vec![7.0, 2.0, 3.0]);
}

#[test]
fn scatter_errors() {
    let x = Tensor::zeros(&[2, 2]);
    let src = Tensor::zeros(&[2, 2]);
    let bad_dim = Tensor::from_vec_i64(vec![5, 0, 0, 0], &[2, 2]).unwrap();
    assert!(x.scatter(1, &bad_dim, &src).is_err());
    let wrong_shape_src = Tensor::zeros(&[3, 2]);
    let ok_idx = Tensor::from_vec_i64(vec![0, 0, 0, 0], &[2, 2]).unwrap();
    assert!(x.scatter(1, &ok_idx, &wrong_shape_src).is_err());
    let float_idx = Tensor::zeros(&[2, 2]);
    assert!(x.scatter(1, &float_idx, &src).is_err());
    let big_idx = Tensor::from_vec_i64(vec![0; 9], &[3, 3]).unwrap();
    let big_src = Tensor::zeros(&[3, 3]);
    assert!(x.scatter(1, &big_idx, &big_src).is_err());
}

#[test]
fn scatter_grad_flows_to_src_at_selected_positions() {
    let x = Tensor::from_vec(vec![0.7, -1.2, 0.5, 0.9], &[2, 2]).unwrap();
    let idx = Tensor::from_vec_i64(vec![0, 1, 1, 0], &[2, 2]).unwrap();
    let src = Tensor::from_vec(vec![0.4, -0.6, 1.1, -0.3], &[2, 2]).unwrap();
    grad_check(&[x.clone(), src.clone()], |t| {
        let i = Tensor::from_vec_i64(vec![0, 1, 1, 0], &[2, 2]).unwrap();
        weighted_loss(t[0].scatter(1, &i, &t[1]).unwrap())
    });
    grad_check(&[src], |t| {
        let i = idx.clone();
        let x = Tensor::from_vec(vec![0.7, -1.2, 0.5, 0.9], &[2, 2]).unwrap();
        weighted_loss(x.scatter(1, &i, &t[0]).unwrap())
    });
}

#[test]
fn scatter_duplicate_writes_grad_exactly() {
    // Both index rows write out[0][0]: k=1 wins, so out = [[7, 2, 3]].
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let src = Tensor::from_vec(vec![5.0, 7.0], &[2, 1])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let idx = Tensor::from_vec_i64(vec![0, 0], &[2, 1]).unwrap();
    let c = Tensor::from_vec(vec![0.3, -1.1, 0.8], &[1, 3]).unwrap();
    base.scatter(0, &idx, &src)
        .unwrap()
        .mul(&c)
        .unwrap()
        .sum()
        .backward();

    // Loser src gets zero; winner takes the full cotangent of its target.
    assert_eq!(src.grad().unwrap().to_vec(), vec![0.0, 0.3]);
    // Written positions carry no flow to self; unwritten pass through.
    assert_eq!(base.grad().unwrap().to_vec(), vec![0.0, -1.1, 0.8]);
}
