use ferro_core::{testkit::grad_check_strict, Tensor};

#[test]
fn singleton_eval_values_stats_and_gradients() {
    for shape in [&[1, 2][..], &[1, 2, 1, 1][..]] {
        let x = Tensor::from_vec(vec![0.7, -0.3], shape).unwrap();
        let w = Tensor::from_vec(vec![1.2, -0.8], &[2]).unwrap();
        let b = Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
        let rm = Tensor::from_vec(vec![0.3, -0.5], &[2]).unwrap();
        let rv = Tensor::from_vec(vec![0.8, 1.4], &[2]).unwrap();
        let out = x.batch_norm(&w, &b, &rm, &rv, 0.01, false, 0.1).unwrap();
        assert_eq!(out.running_mean.to_vec(), rm.to_vec());
        assert_eq!(out.running_var.to_vec(), rv.to_vec());
        for i in 0..2 {
            let expected = (x.to_vec()[i] - rm.to_vec()[i]) / (rv.to_vec()[i] + 0.01).sqrt()
                * w.to_vec()[i]
                + b.to_vec()[i];
            assert!((out.output.to_vec()[i] - expected).abs() < 1e-6);
        }
        grad_check_strict(&[x.clone(), w.clone(), b.clone()], |t| {
            t[0].batch_norm(&t[1], &t[2], &rm, &rv, 0.01, false, 0.1)
                .unwrap()
                .output
                .mul(&Tensor::from_vec(vec![0.3, -0.7], shape).unwrap())
                .unwrap()
                .sum()
        });
        assert!(x.batch_norm(&w, &b, &rm, &rv, 0.01, true, 0.1).is_err());
    }
}

#[test]
fn zero_channels_return_error_not_panic() {
    for shape in [&[2, 0][..], &[2, 0, 3, 4][..]] {
        let x = Tensor::zeros(shape);
        let p = Tensor::zeros(&[0]);
        for train in [false, true] {
            assert!(x.batch_norm(&p, &p, &p, &p, 1e-5, train, 0.1).is_err());
        }
    }
}

#[test]
fn empty_eval_is_empty_with_zero_affine_gradients() {
    for shape in [&[0, 2][..], &[1, 2, 0, 3][..]] {
        let x = Tensor::zeros(shape).requires_grad_(true).unwrap();
        let w = Tensor::ones(&[2]).requires_grad_(true).unwrap();
        let b = Tensor::zeros(&[2]).requires_grad_(true).unwrap();
        let rm = Tensor::zeros(&[2]);
        let rv = Tensor::ones(&[2]);
        let out = x.batch_norm(&w, &b, &rm, &rv, 1e-5, false, 0.1).unwrap();
        assert_eq!(out.output.shape(), shape);
        assert_eq!(out.running_var.to_vec(), vec![1., 1.]);
        out.output.sum().backward();
        assert_eq!(x.grad().unwrap().shape(), shape);
        assert_eq!(w.grad().unwrap().to_vec(), vec![0., 0.]);
        assert_eq!(b.grad().unwrap().to_vec(), vec![0., 0.]);
        assert!(x.batch_norm(&w, &b, &rm, &rv, 1e-5, true, 0.1).is_err());
    }
}

#[test]
fn invalid_dtype_and_nonvector_affine_return_errors() {
    let x = Tensor::zeros(&[2, 2]);
    let p = Tensor::ones(&[2]);
    let bad = Tensor::ones(&[1, 2]).requires_grad_(true).unwrap();
    assert!(x.batch_norm(&bad, &p, &p, &p, 1e-5, false, 0.1).is_err());
    let integer = Tensor::from_vec_i64(vec![0; 4], &[2, 2]).unwrap();
    assert!(integer
        .batch_norm(&p, &p, &p, &p, 1e-5, false, 0.1)
        .is_err());
}
