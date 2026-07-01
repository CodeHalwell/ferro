use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn bmm_values() {
    // batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[10,11]]
    // batch 1: [[1,1,1],[2,2,2]] @ [[1,2],[3,4],[5,6]] = [[9,12],[18,24]]
    let a = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        &[2, 2, 3],
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3, 2],
    )
    .unwrap();
    let c = a.bmm(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2, 2]);
    assert_eq!(c.to_vec(), vec![4.0, 5.0, 10.0, 11.0, 9.0, 12.0, 18.0, 24.0]);
}

#[test]
fn bmm_shape_error() {
    let a = Tensor::from_vec(vec![0.0; 2 * 2 * 3], &[2, 2, 3]).unwrap();
    // inner dim mismatch: (2,2,3) @ (2,4,2)
    let b = Tensor::from_vec(vec![0.0; 2 * 4 * 2], &[2, 4, 2]).unwrap();
    assert!(a.bmm(&b).is_err());
    // batch mismatch: (2,2,3) @ (3,3,2)
    let b2 = Tensor::from_vec(vec![0.0; 3 * 3 * 2], &[3, 3, 2]).unwrap();
    assert!(a.bmm(&b2).is_err());
}

#[test]
fn bmm_grad() {
    let a = Tensor::from_vec(
        vec![0.5, -1.0, 2.0, 1.5, 0.3, -0.7, 1.0, 2.0, -1.0, 0.5, 0.8, -0.2],
        &[2, 2, 3],
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![1.0, -0.5, 0.2, 1.3, -1.0, 0.7, 0.4, 0.9, -0.3, 1.1, 0.6, -0.8],
        &[2, 3, 2],
    )
    .unwrap();
    grad_check(&[a, b], |t| t[0].bmm(&t[1]).unwrap().sum());
}
