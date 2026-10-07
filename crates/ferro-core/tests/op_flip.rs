use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn seq(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec((0..n).map(|i| i as f32).collect(), shape).unwrap()
}

#[test]
fn flip_values() {
    let a = seq(&[2, 3]);
    assert_eq!(a.flip(&[1]).unwrap().to_vec(), vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0]);
    assert_eq!(a.flip(&[0]).unwrap().to_vec(), vec![3.0, 4.0, 5.0, 0.0, 1.0, 2.0]);
    assert_eq!(a.flip(&[0, 1]).unwrap().to_vec(), vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    assert_eq!(a.flip(&[]).unwrap().to_vec(), a.to_vec());
}

#[test]
fn roll_values() {
    let a = seq(&[2, 3]);
    assert_eq!(a.roll(&[1], &[1]).unwrap().to_vec(), vec![2.0, 0.0, 1.0, 5.0, 3.0, 4.0]);
    assert_eq!(a.roll(&[-1], &[1]).unwrap().to_vec(), vec![1.0, 2.0, 0.0, 4.0, 5.0, 3.0]);
    assert_eq!(a.roll(&[4], &[1]).unwrap().to_vec(), a.roll(&[1], &[1]).unwrap().to_vec());
    assert_eq!(a.roll(&[1, 1], &[0, 1]).unwrap().to_vec(), vec![5.0, 3.0, 4.0, 2.0, 0.0, 1.0]);
    let f = a.roll(&[2], &[]).unwrap();
    assert_eq!(f.shape(), &[2, 3]);
    assert_eq!(f.to_vec(), vec![4.0, 5.0, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn flip_roll_errors() {
    let a = seq(&[2, 3]);
    assert!(a.flip(&[2]).is_err());
    assert!(a.flip(&[1, 1]).is_err());
    assert!(a.roll(&[1], &[2]).is_err());
    assert!(a.roll(&[1, 2], &[0]).is_err());
    assert!(a.roll(&[1, 2], &[]).is_err());
}

#[test]
fn flip_roll_grad() {
    let a = Tensor::from_vec((0..6).map(|i| (i as f32 * 0.37).sin()).collect(), &[2, 3]).unwrap();
    let w = Tensor::from_vec(vec![0.5, -1.0, 2.0, 1.5, -0.25, 0.75], &[2, 3]).unwrap();
    let w2 = w.clone();
    grad_check(&[a.clone()], move |t| t[0].flip(&[0, 1]).unwrap().mul(&w).unwrap().sum());
    grad_check(&[a], move |t| t[0].roll(&[1, -2], &[0, 1]).unwrap().mul(&w2).unwrap().sum());
}
