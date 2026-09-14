use ferro_core::{segment, Device, Result, Tensor};
use ferro_core::testkit::grad_check;

#[test]
fn mean_max_and_softmax_forward_backward() {
    let ids = [1,0,1];
    let x = t(&[1.,2.,3.,4.,5.,0.], &[3,2]);
    assert_eq!(segment::mean(&x, &ids, 3).unwrap().to_vec(), vec![3.,4.,3.,1.,0.,0.]);
    assert_eq!(segment::max(&x, &ids, 3).unwrap().to_vec(), vec![3.,4.,5.,2.,f32::NEG_INFINITY,f32::NEG_INFINITY]);
    for op in [segment::mean, segment::max, segment::softmax] {
        grad_check(&[x.detach_copy()], |xs| {
            let y = op(&xs[0], &ids, 2).unwrap();
            y.mul(&y).unwrap().sum()
        });
    }
    let ties = t(&[2.,2.,1.], &[3]).requires_grad_(true).unwrap();
    segment::max(&ties, &[0,0,0], 2).unwrap().sum().backward();
    assert_eq!(ties.grad().unwrap().to_vec(), vec![0.5,0.5,0.]);
    let logits = t(&[1000.,1001.,-1000.], &[3]).requires_grad_(true).unwrap();
    let p = segment::softmax(&logits, &[0,0,1], 3).unwrap();
    let expected = [1. / (1. + 1f32.exp()), 1. / (1. + (-1f32).exp()), 1.];
    for (a,b) in p.to_vec().iter().zip(expected) { assert!((a-b).abs() < 1e-6); }
    p.mul(&t(&[2.,-1.,7.], &[3])).unwrap().sum().backward();
    let dp = logits.grad().unwrap().to_vec();
    assert!((dp[0] - 3. * expected[0] * expected[1]).abs() < 1e-6);
    assert!((dp[1] + dp[0]).abs() < 1e-6);
    assert_eq!(dp[2], 0.);
}

#[test]
fn empty_shapes_validation_and_integer_dtype() {
    for op in [segment::sum, segment::mean, segment::max, segment::softmax] {
        let y = op(&t(&[], &[0,2]), &[], 3).unwrap();
        assert_eq!(y.device(), Device::Cpu);
        assert!(op(&t(&[1.], &[]), &[], 1).is_err());
        assert!(op(&t(&[1.], &[1]), &[1], 1).is_err());
        assert!(op(&t(&[1.], &[1]), &[], 1).is_err());
        assert!(op(&Tensor::from_vec_i64(vec![1], &[1]).unwrap(), &[0], 1).is_err());
        assert_eq!(op(&t(&[], &[2,0]), &[0,0], 1).unwrap().to_vec(), vec![]);
    }
    assert!(segment::softmax(&t(&[f32::INFINITY], &[1]), &[0], 1).is_err());
    assert!(segment::max(&t(&[f32::NAN], &[1]), &[0], 1).is_err());
}

fn t(v: &[f32], shape: &[usize]) -> Tensor { Tensor::from_vec(v.to_vec(), shape).unwrap() }

#[test]
fn unsorted_sum_has_isolated_rows_and_exact_adjoint() {
    let x = t(&[1.,2.,3.,4.,5.,6.], &[3,2]).requires_grad_(true).unwrap();
    let y = segment::sum(&x, &[2,0,2], 4).unwrap();
    assert_eq!(y.to_vec(), vec![3.,4.,0.,0.,6.,8.,0.,0.]);
    y.mul(&t(&[1.,2.,3.,4.,5.,6.,7.,8.], &[4,2])).unwrap().sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![5.,6.,1.,2.,5.,6.]);
    grad_check(&[t(&[0.3,-0.2,0.7,0.4,-0.6,0.9], &[3,2])], |xs| {
        let y = segment::sum(&xs[0], &[2,0,2], 4).unwrap();
        y.mul(&y).unwrap().sum()
    });
}


#[test]
fn mean_and_max_exact_weighted_adjoints() {
    let input = [1.,2.,3.,4.,5.,0.];
    for (op, expected) in [(segment::mean as fn(&Tensor,&[usize],usize)->Result<Tensor>, vec![2.5,3.5,2.,3.,2.5,3.5]),
        (segment::max, vec![0.,7.,2.,3.,5.,0.])] {
        let x = t(&input, &[3,1,2]).requires_grad_(true).unwrap();
        let y = op(&x, &[1,0,1], 3).unwrap();
        y.mul(&t(&[2.,3.,5.,7.,11.,13.], &[3,1,2])).unwrap().sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), expected);
        assert_eq!(x.grad().unwrap().device(), Device::Cpu);
    }
    for v in [f32::INFINITY, f32::NEG_INFINITY] {
        let x = t(&[v,v], &[2]).requires_grad_(true).unwrap();
        segment::max(&x,&[0,0],1).unwrap().sum().backward();
        assert_eq!(x.grad().unwrap().to_vec(), vec![0.5,0.5]);
    }
}

#[test]
fn softmax_matches_independent_f64_reference_on_unsorted_features() {
    let ids = [2,0,2,1,0];
    let data = [0.3,-0.2,0.6,0.9,-0.4,0.8,0.5,-0.7,0.1,0.2];
    let upstream = [0.2,0.4,0.1,-0.3,0.8,0.6,-0.2,0.7,0.5,0.9];
    let x = t(&data, &[5,2]).requires_grad_(true).unwrap();
    let y = segment::softmax(&x,&ids,4).unwrap();
    let mut p = vec![0f64;10];
    let mut dx = vec![0f64;10];
    for group in 0..4 { for f in 0..2 {
        let members: Vec<_> = (0..5).filter(|&e| ids[e] == group).collect();
        let z: f64 = members.iter().map(|&e| (data[e*2+f] as f64).exp()).sum();
        for &e in &members { p[e*2+f] = (data[e*2+f] as f64).exp()/z; }
        let dot: f64 = members.iter().map(|&e| p[e*2+f]*upstream[e*2+f] as f64).sum();
        for &e in &members { dx[e*2+f] = p[e*2+f]*(upstream[e*2+f] as f64-dot); }
    } }
    y.mul(&t(&upstream, &[5,2])).unwrap().sum().backward();
    for (a,b) in y.to_vec().iter().zip(p) { assert!((*a as f64-b).abs() < 1e-7); }
    for (a,b) in x.grad().unwrap().to_vec().iter().zip(dx) { assert!((*a as f64-b).abs() < 1e-7); }
}

#[test]
fn noncontiguous_segment_input_retains_logical_feature_order() {
    let base = t(&[1.,3.,5.,2.,4.,6.], &[2,3]).requires_grad_(true).unwrap();
    let x = base.transpose(0,1).unwrap();
    let y = segment::sum(&x,&[1,0,1],2).unwrap();
    assert_eq!(y.to_vec(),vec![3.,4.,6.,8.]);
    y.mul(&t(&[1.,2.,3.,4.], &[2,2])).unwrap().sum().backward();
    assert_eq!(base.grad().unwrap().to_vec(),vec![3.,1.,3.,4.,2.,4.]);
}
