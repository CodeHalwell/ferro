use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.16 + 0.19 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn pad_constant_values() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    // One before / zero after on dim 0; zero before / two after on dim 1.
    let y = a.pad_constant(&[1, 0, 0, 2], -1.0).unwrap();
    assert_eq!(y.shape(), &[3, 4]);
    // Row 0 is the dim-0 front pad; input rows keep values plus two dim-1 pads.
    assert_eq!(
        y.to_vec(),
        vec![-1.0, -1.0, -1.0, -1.0, 1.0, 2.0, -1.0, -1.0, 3.0, 4.0, -1.0, -1.0]
    );

    // No-op padding returns the input values.
    let z = a.pad_constant(&[0, 0, 0, 0], 7.0).unwrap();
    assert_eq!(z.to_vec(), a.to_vec());

    // 1-D padding.
    let v = Tensor::from_vec(vec![5.0, 6.0], &[2]).unwrap();
    let p = v.pad_constant(&[1, 1], 0.0).unwrap();
    assert_eq!(p.shape(), &[4]);
    assert_eq!(p.to_vec(), vec![0.0, 5.0, 6.0, 0.0]);
}

#[test]
fn pad_constant_errors() {
    let a = Tensor::zeros(&[2, 2]);
    assert!(a.pad_constant(&[1, 1], 0.0).is_err()); // wrong pad count
}

#[test]
fn pad_constant_grad() {
    let a = Tensor::from_vec(vec![0.7, -1.1, 0.4, 0.9], &[2, 2]).unwrap();
    grad_check(&[a.clone()], |t| {
        weighted_loss(t[0].pad_constant(&[1, 0, 0, 2], -1.0).unwrap())
    });

    let v = Tensor::from_vec(vec![0.5, -0.6, 1.2], &[3]).unwrap();
    grad_check(&[v], |t| {
        weighted_loss(t[0].pad_constant(&[2, 1], 3.0).unwrap())
    });
}
