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
fn scatter_add_values() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let idx = Tensor::from_vec_i64(vec![1, 0, 1, 0], &[2, 2]).unwrap();
    let src = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();

    // out[i][idx[i][j]][j] = self[i][idx[i][j]][j] + src[i][j]:
    // row 0 -> [self[0][0]+20, self[0][1]+10] = [21, 12]
    // row 1 -> [self[1][0]+40, self[1][1]+30] = [43, 34]
    let y = x.scatter_add(1, &idx, &src).unwrap();
    assert_eq!(y.to_vec(), vec![21.0, 12.0, 43.0, 34.0]);

    // Duplicate indices accumulate.
    let dup_idx = Tensor::from_vec_i64(vec![0, 0], &[1, 2]).unwrap();
    let dup_src = Tensor::from_vec(vec![5.0, 7.0], &[1, 2]).unwrap();
    let base = Tensor::zeros(&[1, 3]);
    let z = base.scatter_add(1, &dup_idx, &dup_src).unwrap();
    assert_eq!(z.to_vec(), vec![12.0, 0.0, 0.0]);
}

#[test]
fn scatter_add_errors() {
    let x = Tensor::zeros(&[2, 2]);
    let src = Tensor::zeros(&[2, 2]);
    let bad_dim = Tensor::from_vec_i64(vec![5, 0, 0, 0], &[2, 2]).unwrap();
    assert!(x.scatter_add(1, &bad_dim, &src).is_err());
    let wrong_shape_src = Tensor::zeros(&[3, 2]);
    let ok_idx = Tensor::from_vec_i64(vec![0, 0, 0, 0], &[2, 2]).unwrap();
    assert!(x.scatter_add(1, &ok_idx, &wrong_shape_src).is_err());
    // index larger than input outside dim is rejected
    let big_idx = Tensor::from_vec_i64(vec![0; 9], &[3, 3]).unwrap();
    let big_src = Tensor::zeros(&[3, 3]);
    assert!(x.scatter_add(1, &big_idx, &big_src).is_err());
}

#[test]
fn scatter_add_grad() {
    let x = Tensor::from_vec(vec![0.7, -1.2, 0.5, 0.9], &[2, 2]).unwrap();
    let idx = Tensor::from_vec_i64(vec![1, 0, 1, 0], &[2, 2]).unwrap();
    let src = Tensor::from_vec(vec![0.4, -0.6, 1.1, -0.3], &[2, 2]).unwrap();
    grad_check(&[x.clone(), src.clone()], |t| {
        let i = Tensor::from_vec_i64(vec![1, 0, 1, 0], &[2, 2]).unwrap();
        weighted_loss(t[0].scatter_add(1, &i, &t[1]).unwrap())
    });
    grad_check(&[x], |t| {
        let i = idx.clone();
        let s = Tensor::from_vec(vec![0.4, -0.6, 1.1, -0.3], &[2, 2]).unwrap();
        weighted_loss(t[0].scatter_add(1, &i, &s).unwrap())
    });
}
