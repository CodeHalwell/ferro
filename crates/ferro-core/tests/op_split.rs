use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn seq(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec((0..n).map(|i| i as f32).collect(), shape).unwrap()
}

#[test]
fn split_and_chunk_values() {
    let a = seq(&[2, 5]);
    let parts = a.split(2, 1).unwrap();
    let shapes: Vec<_> = parts.iter().map(|p| p.shape().to_vec()).collect();
    assert_eq!(shapes, vec![vec![2, 2], vec![2, 2], vec![2, 1]]);
    assert_eq!(parts[2].to_vec(), vec![4.0, 9.0]);
    assert!(parts.iter().all(|p| p._storage_ptr() == a._storage_ptr()));

    let s = a.split_sizes(&[1, 4], 1).unwrap();
    assert_eq!(s[0].to_vec(), vec![0.0, 5.0]);
    assert_eq!(s[1].to_vec(), vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]);

    // chunk(3) of 5 -> sizes 2,2,1; chunk(4) of 6 -> size 2 x 3 (fewer than n).
    assert_eq!(a.chunk(3, 1).unwrap().len(), 3);
    assert_eq!(seq(&[6]).chunk(4, 0).unwrap().len(), 3);
    assert_eq!(a.chunk(2, 0).unwrap()[1].to_vec(), vec![5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn split_errors() {
    let a = seq(&[2, 5]);
    assert!(a.split(0, 1).is_err());
    assert!(a.split(1, 2).is_err());
    assert!(a.split_sizes(&[2, 2], 1).is_err());
    assert!(a.split_sizes(&[usize::MAX, 6], 1).is_err());
    assert!(a.chunk(0, 1).is_err());
}

#[test]
fn split_grad() {
    let a = Tensor::from_vec((0..10).map(|i| (i as f32 * 0.37).sin()).collect(), &[2, 5]).unwrap();
    grad_check(&[a.clone()], |t| {
        let p = t[0].split(2, 1).unwrap();
        p[0].mul(&p[1]).unwrap().sum().add(&p[2].exp().sum()).unwrap()
    });
    grad_check(&[a], |t| {
        let c = t[0].chunk(2, 0).unwrap();
        c[0].mul(&c[1]).unwrap().sum()
    });
}
