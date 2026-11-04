use ferro_core::{DType, Tensor};

fn values(dtype: DType) -> Tensor {
    Tensor::from_vec_f64((1..=24).map(|n| n as f64).collect(), &[4, 6]).unwrap().to_dtype(dtype)
}

#[test]
fn initial_and_repeated_leaf_gradients_follow_dense_or_contiguous_layouts() {
    for dtype in [DType::F32, DType::F64] {
        let base = values(dtype);
        let cases = [
            (base.clone(), vec![6, 1]),
            (base.transpose(0, 1).unwrap(), vec![1, 6]),
            (base.slice(1, 0, 6, 2).unwrap(), vec![3, 1]),
            (base.narrow(0, 0, 1).unwrap().expand(&[3, 6]).unwrap(), vec![6, 1]),
            (base.narrow(0, 1, 2).unwrap(), vec![6, 1]),
            (base.narrow(0, 0, 0).unwrap().narrow(1, 3, 2).unwrap(), vec![6, 1]),
        ];
        for (view, stride) in cases {
            let leaf = view.detach().requires_grad_(true).unwrap();
            let expected = leaf.to_vec_f64().into_iter().map(|v| 2. * v).collect::<Vec<_>>();
            let version = leaf._version();
            let storage = leaf._storage_ptr();
            let loss = leaf.transpose(0, 1).unwrap().square().unwrap().sum();
            loss.backward();
            let held = leaf.grad().unwrap();
            assert_eq!(held.strides(), stride);
            assert_eq!(held.storage_offset(), 0);
            assert_eq!(held.dtype(), dtype);
            assert_eq!(held.to_vec_f64(), expected);
            assert_ne!(held._storage_ptr(), storage);
            loss.backward();
            let repeated = leaf.grad().unwrap();
            assert_eq!(repeated.strides(), stride);
            assert_eq!(repeated.to_vec_f64(), expected.iter().map(|v| v * 2.).collect::<Vec<_>>());
            assert_eq!(held.id(), repeated.id());
            assert_eq!(held.to_vec_f64(), repeated.to_vec_f64());
            assert_eq!(leaf._storage_ptr(), storage);
            assert_eq!(leaf._version(), version);
        }
    }
}

#[test]
fn assigned_dense_gradient_keeps_its_layout_and_float64_precision() {
    let leaf = Tensor::from_vec_f64(vec![1e100, 1.0000000000000002, -2e100, -2.], &[2, 2]).unwrap().requires_grad_(true).unwrap();
    let initial = Tensor::from_vec_f64(vec![1e-200, 2., 3., 4.], &[2, 2]).unwrap().transpose(0, 1).unwrap();
    leaf.set_leaf_grad(Some(initial.clone())).unwrap();
    assert_eq!(leaf.grad().unwrap().id(), initial.id());
    let loss = leaf.square().unwrap().sum();
    loss.backward();
    let result = leaf.grad().unwrap();
    assert_eq!(result.strides(), initial.strides());
    assert_eq!(result.to_vec_f64(), vec![2e100, 5., -4e100, 0.]);
    assert_eq!(initial.id(), result.id());
    assert_eq!(initial.to_vec_f64(), result.to_vec_f64());
}
