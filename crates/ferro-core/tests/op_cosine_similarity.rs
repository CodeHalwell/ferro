use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn cosine_similarity_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let same = a.cosine_similarity(&a, 0, 1e-8).unwrap().item();
    assert!((same - 1.0).abs() < 1e-5);

    let b = Tensor::from_vec(vec![1.0, 0.0], &[2]).unwrap();
    let c = Tensor::from_vec(vec![0.0, 1.0], &[2]).unwrap();
    let orth = b.cosine_similarity(&c, 0, 1e-8).unwrap().item();
    assert!(orth.abs() < 1e-5);
}

#[test]
fn cosine_similarity_out_of_range_is_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    assert!(a.cosine_similarity(&a, 1, 1e-8).is_err());
}

#[test]
fn cosine_similarity_grad() {
    let a = Tensor::from_vec(vec![1.0, 2.0, -0.5, 0.7], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![0.3, -1.2, 2.0, 1.1], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| {
        t[0].cosine_similarity(&t[1], 1, 1e-8).unwrap().sum()
    });
}
