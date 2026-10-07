use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn seq(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec((0..n).map(|i| i as f32).collect(), shape).unwrap()
}

#[test]
fn slice_and_narrow_values() {
    let a = seq(&[3, 5]);
    let s = a.slice(1, 1, 5, 2).unwrap();
    assert_eq!(s.shape(), &[3, 2]);
    assert_eq!(s.to_vec(), vec![1.0, 3.0, 6.0, 8.0, 11.0, 13.0]);
    assert_eq!(s._storage_ptr(), a._storage_ptr());

    let n = a.narrow(0, 1, 2).unwrap();
    assert_eq!(n.shape(), &[2, 5]);
    assert_eq!(n.to_vec(), (5..15).map(|i| i as f32).collect::<Vec<_>>());

    // Python-style clamping, empty ranges and views of views.
    assert_eq!(a.slice(1, 3, 100, 1).unwrap().shape(), &[3, 2]);
    let e = a.slice(1, 5, 5, 1).unwrap();
    assert_eq!(e.shape(), &[3, 0]);
    assert!(e.to_vec().is_empty());
    let vv = a.slice(1, 0, 5, 2).unwrap().narrow(0, 2, 1).unwrap();
    assert_eq!(vv.to_vec(), vec![10.0, 12.0, 14.0]);
    let t = a.transpose(0, 1).unwrap().narrow(0, 4, 1).unwrap();
    assert_eq!(t.to_vec(), vec![4.0, 9.0, 14.0]);
    let i = Tensor::from_vec_i64(vec![1, 2, 3, 4], &[4]).unwrap();
    assert_eq!(i.slice(0, 1, 4, 2).unwrap().to_vec_i64(), vec![2, 4]);
}

#[test]
fn prefix_view_feeds_downstream_ops() {
    // offset 0, contiguous, but shorter than storage: must not be treated as
    // the whole buffer by any fast path.
    let a = seq(&[4, 3]);
    let p = a.narrow(0, 0, 1).unwrap();
    assert_eq!(p.exp().to_vec(), vec![1.0, 1f32.exp(), 2f32.exp()]);
    assert_eq!(p.add(&p).unwrap().to_vec(), vec![0.0, 2.0, 4.0]);
    assert_eq!(p.sum().item(), 3.0);
    assert_eq!(p.reshape(&[3]).unwrap().to_vec(), vec![0.0, 1.0, 2.0]);
}

#[test]
fn slice_errors() {
    let a = seq(&[3, 5]);
    assert!(a.slice(2, 0, 1, 1).is_err());
    assert!(a.slice(0, 0, 1, 0).is_err());
    assert!(a.narrow(1, 3, 3).is_err());
    assert!(a.narrow(1, usize::MAX, 2).is_err());
    assert!(a.narrow(2, 0, 1).is_err());
}

#[test]
fn slice_grad() {
    let a = Tensor::from_vec((0..15).map(|i| (i as f32 * 0.37).sin()).collect(), &[3, 5]).unwrap();
    let w = Tensor::from_vec(vec![0.5, -1.0, 2.0, 1.5, -0.25, 0.75], &[3, 2]).unwrap();
    grad_check(&[a.clone()], move |t| t[0].slice(1, 1, 5, 2).unwrap().mul(&w).unwrap().sum());
    let w2 = Tensor::from_vec((0..10).map(|i| i as f32 * 0.1 - 0.4).collect(), &[2, 5]).unwrap();
    grad_check(&[a], move |t| t[0].narrow(0, 1, 2).unwrap().mul(&w2).unwrap().sum());
}
