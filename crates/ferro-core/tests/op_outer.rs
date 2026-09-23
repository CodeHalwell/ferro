use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec((0..n).map(|i| 0.15 + 0.27 * i as f32).collect::<Vec<_>>(), y.shape()).unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn outer_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0], &[2]).unwrap();
    let y = a.outer(&b).unwrap();
    assert_eq!(y.shape(), &[3, 2]);
    assert_eq!(y.to_vec(), vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
}

#[test]
fn outer_rejects_non_vectors() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0], &[2]).unwrap();
    assert!(a.outer(&b).is_err());
    assert!(b.outer(&a).is_err());
    assert!(Tensor::scalar(1.0).outer(&b).is_err());
}

#[test]
fn outer_grad() {
    let a = Tensor::from_vec(vec![0.3, -1.2, 2.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![1.5, -0.7], &[2]).unwrap();
    grad_check(&[a, b], |t| weighted_loss(t[0].outer(&t[1]).unwrap()));
}
