use ferro_core::testkit::grad_check;
use ferro_core::Tensor;

#[test]
fn hinge_embedding_loss_values() {
    let margin = 1.0f32;
    let x = Tensor::from_vec(vec![0.5, 2.0, 0.3, 3.0], &[4]).unwrap();
    let target = Tensor::from_vec(vec![1.0, 1.0, -1.0, -1.0], &[4]).unwrap();
    let got = x.hinge_embedding_loss(&target, margin).unwrap().to_vec();
    // losses = [0.5, 2.0, max(0,0.7)=0.7, max(0,-2.0)=0.0]; mean = 3.2/4 = 0.8
    assert!((got[0] - 0.8).abs() < 1e-5);
}

#[test]
fn hinge_embedding_loss_grad() {
    let margin = 1.0f32;
    let a = Tensor::from_vec(vec![0.3, -0.2, 2.5, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![1.0, -1.0, 1.0, -1.0], &[2, 2]).unwrap();
    grad_check(&[a, b], |t| {
        t[0].hinge_embedding_loss(&t[1], margin).unwrap()
    });
}
