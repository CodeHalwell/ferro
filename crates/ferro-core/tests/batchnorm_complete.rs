use ferro_core::{testkit::grad_check_strict, Tensor};

#[test]
fn rank_three_and_five_training_values_and_all_gradients() {
    for shape in [vec![2, 2, 3], vec![2, 2, 2, 1, 2]] {
        let count = shape.iter().product();
        let x = Tensor::from_vec((0..count).map(|i| (i as f32 * 0.7).sin()).collect(), &shape).unwrap();
        let w = Tensor::from_vec(vec![0.8, -1.2], &[2]).unwrap();
        let b = Tensor::from_vec(vec![0.2, 0.5], &[2]).unwrap();
        let rm = Tensor::zeros(&[2]);
        let rv = Tensor::ones(&[2]);
        let probe = Tensor::from_vec((0..count).map(|i| (i as f32 * 0.3).cos()).collect(), &shape).unwrap();
        grad_check_strict(&[x, w, b], |a| {
            a[0].batch_norm(&a[1], &a[2], &rm, &rv, 0.03, true, 0.2).unwrap().output.mul(&probe).unwrap().sum()
        });
    }
}

#[test]
fn zero_variance_tiny_epsilon_has_finite_correct_gradients() {
    let x = Tensor::ones(&[2, 1]).requires_grad_(true).unwrap();
    let one = Tensor::ones(&[1]);
    let zero = Tensor::zeros(&[1]);
    let y = x.batch_norm(&one, &zero, &zero, &one, 1e-40, true, 0.1).unwrap().output;
    assert_eq!(y.to_vec(), vec![0.0; 2]);
    y.mul(&Tensor::from_vec(vec![1.0, -1.0], &[2, 1]).unwrap()).unwrap().sum().backward();
    let expected = 1.0 / 1e-40_f32.sqrt();
    for (value, sign) in x.grad().unwrap().to_vec().iter().zip([1.0, -1.0]) {
        assert!(value.is_finite());
        assert!((*value / expected - sign).abs() < 1e-6);
    }
}

#[test]
fn enormous_empty_batches_do_not_loop_or_alias_returned_statistics() {
    let shape = [usize::MAX / 4, 2, 0];
    let x = Tensor::zeros(&shape).requires_grad_(true).unwrap();
    let w = Tensor::ones(&[2]).requires_grad_(true).unwrap();
    let b = Tensor::zeros(&[2]).requires_grad_(true).unwrap();
    let rm = Tensor::zeros(&[2]);
    let rv = Tensor::ones(&[2]);
    let result = x.batch_norm(&w, &b, &rm, &rv, 1e-5, true, 0.1).unwrap();
    assert_eq!(result.output.shape(), &shape);
    result.output.sum().backward();
    assert_eq!(x.grad().unwrap().numel(), 0);
    assert_eq!(w.grad().unwrap().to_vec(), vec![0.0; 2]);
    result.running_mean.copy_from(&Tensor::ones(&[2])).unwrap();
    assert_eq!(rm.to_vec(), vec![0.0; 2]);
}

#[test]
fn eval_checks_saved_running_statistics_and_rejects_their_gradients() {
    let x = Tensor::ones(&[2, 2]).requires_grad_(true).unwrap();
    let w = Tensor::ones(&[2]);
    let b = Tensor::zeros(&[2]);
    let rm = Tensor::zeros(&[2]);
    let rv = Tensor::ones(&[2]);
    let y = x.batch_norm(&w, &b, &rm, &rv, 1e-5, false, 0.1).unwrap().output;
    rm.copy_from(&Tensor::ones(&[2])).unwrap();
    assert!(y.sum().grad_wrt(&[&x], false).is_err());
    assert!(x.grad().is_none());
    let live = rv.requires_grad_(true).unwrap();
    assert!(x.batch_norm(&w, &b, &rm, &live, 1e-5, false, 0.1).is_err());
}
