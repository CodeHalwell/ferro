use ferro_core::testkit::grad_check;
use ferro_core::{DType, Tensor};

fn seq(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec((0..n).map(|i| i as f32).collect(), shape).unwrap()
}

#[test]
fn permute_values_and_view() {
    let a = seq(&[2, 3, 4]);
    let p = a.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.shape(), &[4, 2, 3]);
    let (src, out) = (a.to_vec(), p.to_vec());
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(out[(i * 2 + j) * 3 + k], src[(j * 3 + k) * 4 + i]);
            }
        }
    }
    assert_eq!(p._storage_ptr(), a._storage_ptr());
    assert_eq!(a.permute(&[0, 1, 2]).unwrap().to_vec(), src);
    let i = Tensor::from_vec_i64(vec![1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let pi = i.permute(&[1, 0]).unwrap();
    assert_eq!(pi.dtype(), DType::I64);
    assert_eq!(pi.to_vec_i64(), vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn permute_errors() {
    let a = seq(&[2, 3]);
    assert!(a.permute(&[0]).is_err());
    assert!(a.permute(&[0, 0]).is_err());
    assert!(a.permute(&[0, 2]).is_err());
}

#[test]
fn permute_grad() {
    let a = Tensor::from_vec((0..24).map(|i| (i as f32 * 0.37).sin()).collect(), &[2, 3, 4]).unwrap();
    let w = Tensor::from_vec((0..24).map(|i| (i as f32 * 0.91).cos()).collect(), &[4, 2, 3]).unwrap();
    grad_check(&[a], move |t| t[0].permute(&[2, 0, 1]).unwrap().mul(&w).unwrap().sum());
}
