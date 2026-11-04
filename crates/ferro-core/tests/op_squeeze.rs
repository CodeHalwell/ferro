use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn unsqueeze_values() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::from_vec(data.clone(), &[2, 3]).unwrap();
    for (dim, shape) in [(0, [1, 2, 3]), (1, [2, 1, 3]), (2, [2, 3, 1])] {
        let out = a.unsqueeze(dim).unwrap();
        assert_eq!(out.shape(), &shape);
        assert_eq!(out.to_vec(), data);
    }
}

#[test]
fn squeeze_values() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::from_vec(data.clone(), &[1, 2, 3]).unwrap();
    let out = a.squeeze(0).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.to_vec(), data);

    let b = Tensor::from_vec(data.clone(), &[2, 1, 3]).unwrap();
    let out = b.squeeze(1).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.to_vec(), data);

    assert!(b.squeeze(0).is_err());
    assert!(b.squeeze(3).is_err());
    assert!(b.unsqueeze(4).is_err());
}

#[test]
fn squeeze_roundtrip_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.5, 2.0, 3.0, -0.25, 1.0], &[2, 3]).unwrap();
    let w = Tensor::from_vec(vec![1.5, -2.0, 0.5, 1.0, 2.5, -1.0], &[2, 3]).unwrap();
    grad_check(&[a], |t| {
        t[0].unsqueeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .mul(&w)
            .unwrap()
            .sum()
    });
}

#[test]
fn singleton_views_preserve_strided_storage_aliases_and_typed_adjoints() {
    let storage = Tensor::from_vec_f64(vec![1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
    let leaf = storage.requires_grad_(true).unwrap();
    let transposed = leaf.transpose(0, 1).unwrap();
    let inserted = transposed.unsqueeze(1).unwrap();
    let removed = inserted.squeeze(1).unwrap();
    removed.square().unwrap().sum().backward();
    assert_eq!(leaf.grad().unwrap().to_vec_f64(), vec![2., 4., 6., 8., 10., 12.]);
    let stale = inserted.square().unwrap().sum();
    storage.copy_from(&Tensor::from_vec_f64(vec![7., 8., 9., 10., 11., 12.], &[2, 3]).unwrap()).unwrap();
    assert_eq!(inserted.to_vec_f64(), vec![7., 10., 8., 11., 9., 12.]);
    assert_eq!(removed.to_vec_f64(), vec![7., 10., 8., 11., 9., 12.]);
    assert!(stale.grad_wrt(&[&leaf], false).is_err());
    let boolean = Tensor::from_vec_bool(vec![true, false, true, false], &[2, 2]).unwrap();
    let view = boolean.transpose(0, 1).unwrap().unsqueeze(2).unwrap();
    boolean.copy_from(&Tensor::from_vec_bool(vec![false, true, false, true], &[2, 2]).unwrap()).unwrap();
    assert_eq!(view.squeeze(2).unwrap().to_vec_bool(), vec![false, false, true, true]);
}

#[test]
fn singleton_view_capture_replay_reads_updated_leaf_storage() {
    let input = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
    let output = ferro_core::capture(|| input.unsqueeze(0).unwrap().squeeze(0).unwrap().mul(&input).unwrap());
    let chain = ferro_core::graph::CompiledChain::compile(&output).unwrap();
    input.copy_from(&Tensor::from_vec(vec![2., 3., 4., 5.], &[2, 2]).unwrap()).unwrap();
    assert_eq!(chain.replay().unwrap().to_vec(), vec![4., 9., 16., 25.]);
}
