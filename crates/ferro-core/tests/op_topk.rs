use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn topk_values_and_indices() {
    let a = Tensor::from_vec(vec![1.0, 4.0, 2.0, 5.0, 0.0, 3.0], &[2, 3]).unwrap();
    let (v, i) = a.topk(2, 1).unwrap();
    assert_eq!(v.shape(), &[2, 2]);
    assert_eq!(v.to_vec(), vec![4.0, 2.0, 5.0, 3.0]);
    assert_eq!(i.to_vec_i64(), vec![1, 2, 0, 2]);
}

#[test]
fn topk_along_dim0() {
    let a = Tensor::from_vec(vec![1.0, 4.0, 2.0, 5.0, 0.0, 3.0], &[2, 3]).unwrap();
    let (v, i) = a.topk(1, 0).unwrap();
    assert_eq!(v.shape(), &[1, 3]);
    assert_eq!(v.to_vec(), vec![5.0, 4.0, 3.0]);
    assert_eq!(i.to_vec_i64(), vec![1, 0, 1]);
}

#[test]
fn topk_ties_take_lowest_index() {
    let a = Tensor::from_vec(vec![2.0, 3.0, 2.0], &[3]).unwrap();
    let (v, i) = a.topk(3, 0).unwrap();
    assert_eq!(v.to_vec(), vec![3.0, 2.0, 2.0]);
    assert_eq!(i.to_vec_i64(), vec![1, 0, 2]);
}

#[test]
fn topk_rejects_bad_args() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    assert!(a.topk(3, 0).is_err());
    assert!(a.topk(1, 1).is_err());
}

#[test]
fn topk_grad() {
    let a = Tensor::from_vec(vec![0.9, -0.4, 1.6, 0.1, -1.2, 0.7], &[2, 3]).unwrap();
    grad_check(&[a], |t| {
        let (v, _) = t[0].topk(2, 1).unwrap();
        v.mul(&v).unwrap().sum()
    });
}

#[test]
fn empty_topk_skips_huge_outer_loops_and_source_materialization() {
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for (shape, k, dim) in [(vec![1usize << 60, 0], 0, 1), (vec![1usize << 60, 4, 0], 1, 1), (vec![0, 1usize << 60], 0, 1)] {
            let a = Tensor::from_vec(Vec::new(), &shape).unwrap().requires_grad_(true).unwrap();
            let (values, indices) = a.topk(k, dim).unwrap();
            let mut expected = shape.clone(); expected[dim] = k;
            assert_eq!(values.shape(), expected);
            assert_eq!(indices.shape(), expected);
            assert_eq!(indices.dtype(), ferro_core::DType::I64);
            assert!(!indices.requires_grad());
            values.sum().backward();
            assert_eq!(a.grad().unwrap().shape(), shape);
            assert!(a.grad().unwrap().to_vec().is_empty());
        }
        let huge = Tensor::scalar(1.).expand(&[1usize << 60]).unwrap();
        let (values, indices) = huge.topk(0, 0).unwrap();
        assert_eq!(values.shape(), &[0]);
        assert_eq!(indices.shape(), &[0]);
        send.send(()).unwrap();
    });
    receive.recv_timeout(std::time::Duration::from_secs(2)).expect("empty topk must not iterate outer groups or materialize its source");
}

#[test]
fn empty_topk_preserves_zero_gradients_and_capture_barrier() {
    let a = Tensor::ones(&[3, 4]).requires_grad_(true).unwrap();
    let (values, indices) = a.topk(0, 1).unwrap();
    values.sum().backward();
    assert_eq!(a.grad().unwrap().to_vec(), vec![0.; 12]);
    assert_eq!(indices.shape(), &[3, 0]);
    let input = Tensor::zeros(&[1usize << 60, 0]);
    let (values, indices) = ferro_core::capture(|| input.topk(0, 1).unwrap());
    assert_eq!(values.shape(), input.shape());
    assert!(ferro_core::graph::Graph::from_root(&values).nodes.len() > 1);
    assert!(!indices.requires_grad());
}

#[test]
fn empty_topk_keeps_dtype_and_argument_validation() {
    let shape = &[1usize << 60, 0];
    for input in [Tensor::from_vec_f64(Vec::new(), shape).unwrap(), Tensor::from_vec_i64(Vec::new(), shape).unwrap(), Tensor::from_vec_bool(Vec::new(), shape).unwrap()] {
        assert!(input.topk(0, 1).is_err());
    }
    let a = Tensor::zeros(shape);
    assert!(a.topk(1, 1).is_err());
    assert!(a.topk(0, 2).is_err());
}
