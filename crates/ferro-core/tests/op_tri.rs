use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.12 + 0.41 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn triu_values() {
    let a = Tensor::from_vec((1..=9).map(|v| v as f32).collect(), &[3, 3]).unwrap();
    assert_eq!(
        a.triu(0).unwrap().to_vec(),
        vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
    );
    // diagonal=1 excludes the main diagonal; diagonal=-1 keeps the sub-diagonal.
    assert_eq!(
        a.triu(1).unwrap().to_vec(),
        vec![0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        a.triu(-1).unwrap().to_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]
    );
}

#[test]
fn tril_values() {
    let a = Tensor::from_vec((1..=9).map(|v| v as f32).collect(), &[3, 3]).unwrap();
    assert_eq!(
        a.tril(0).unwrap().to_vec(),
        vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(
        a.tril(1).unwrap().to_vec(),
        vec![1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(
        a.tril(-1).unwrap().to_vec(),
        vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0]
    );
}

#[test]
fn tri_errors() {
    let v = Tensor::zeros(&[4]);
    assert!(v.triu(0).is_err());
    assert!(v.tril(0).is_err());
    let m = Tensor::zeros(&[2, 2]);
    assert!(m.triu(0).is_ok());
}

#[test]
fn tri_grad() {
    let a = Tensor::from_vec(
        vec![0.7, -1.2, 0.5, 0.9, -0.3, 1.1, 0.4, -0.8, 0.6],
        &[3, 3],
    )
    .unwrap();
    grad_check(&[a.clone()], |t| weighted_loss(t[0].triu(0).unwrap()));
    grad_check(&[a.clone()], |t| weighted_loss(t[0].triu(1).unwrap()));
    grad_check(&[a.clone()], |t| weighted_loss(t[0].tril(-1).unwrap()));

    // Rectangular matrix with negative diagonal offset.
    let r = Tensor::from_vec(vec![0.5, -0.9, 0.3, 1.0, -0.2, 0.8], &[2, 3]).unwrap();
    grad_check(&[r], |t| weighted_loss(t[0].tril(0).unwrap()));
}
