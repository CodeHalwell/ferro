use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

fn weighted_loss(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.19 + 0.21 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn logsumexp_matches_manual() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
    let y = a.logsumexp(1).unwrap();
    assert_eq!(y.shape(), &[1]);
    // sum(exp(x)) = e + e^2 + e^3; lse = ln(that).
    let want = (1f32.exp() + 2f32.exp() + 3f32.exp()).ln();
    assert!((y.item() - want).abs() < 1e-4);
}

#[test]
fn logsumexp_dim_shapes() {
    let a = Tensor::randn(&[2, 3, 4], &ferro_core::Rng::new(5));
    assert_eq!(a.logsumexp(0).unwrap().shape(), &[3, 4]);
    assert_eq!(a.logsumexp(2).unwrap().shape(), &[2, 3]);
    assert!(a.logsumexp(3).is_err());

    // exp(lse(x, dim=1)) at [r, j] equals sum_k exp(x[r][k][j]).
    let lse = a.logsumexp(1).unwrap().to_vec();
    let xv = a.to_vec();
    for r in 0..2 {
        for j in 0..4 {
            let want: f32 = (0..3).map(|k| xv[(r * 3 + k) * 4 + j].exp()).sum();
            assert!((lse[r * 4 + j].exp() - want).abs() / want.abs() < 1e-4);
        }
    }
}

#[test]
fn logsumexp_grad() {
    let a = Tensor::from_vec(vec![0.5, -1.0, 0.3, 1.2, 0.1, -0.4], &[2, 3]).unwrap();
    grad_check(&[a.clone()], |t| weighted_loss(t[0].logsumexp(1).unwrap()));
    grad_check(&[a.clone()], |t| weighted_loss(t[0].logsumexp(0).unwrap()));

    // Gradient of logsumexp is the softmax; verify against softmax op.
    let g = {
        let leaf = a.requires_grad_(true).unwrap();
        leaf.logsumexp(1).unwrap().sum().backward();
        leaf.grad().unwrap().to_vec()
    };
    let sm = a.softmax(1).unwrap().to_vec();
    for (gg, ss) in g.iter().zip(&sm) {
        assert!((gg - *ss).abs() < 1e-5);
    }
}
