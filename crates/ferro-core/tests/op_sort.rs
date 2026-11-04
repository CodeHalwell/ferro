use ferro_core::{Device, Tensor};
use ferro_core::testkit::grad_check;

#[test]
fn typed_axis_sort_keeps_exact_values_and_stable_ties() {
    let x = Tensor::from_vec(vec![3., 1., 2., 0., 5., 4.], &[2,3]).unwrap();
    let (v, i) = x.sort(1, false, true).unwrap();
    assert_eq!(v.to_vec(), vec![1.,2.,3.,0.,4.,5.]);
    assert_eq!(i.to_vec_i64(), vec![1,2,0,0,2,1]);
    assert_eq!(x.sort(0,true,true).unwrap().0.to_vec(),vec![3.,5.,4.,0.,1.,2.]);
    let n=1i64<<62;
    let x=Tensor::from_vec_i64(vec![n+3,n+1,n+2,n+1],&[4]).unwrap();
    let (v,i)=x.sort(0,false,true).unwrap();
    assert_eq!(v.to_vec_i64(),vec![n+1,n+1,n+2,n+3]);
    assert_eq!(i.to_vec_i64(),vec![1,3,2,0]);
    let x=Tensor::from_vec_bool(vec![true,false,true,false],&[4]).unwrap();
    assert_eq!(x.argsort(0,false,true).unwrap().to_vec_i64(),vec![1,3,0,2]);
    for descending in [false,true] {
        let x=Tensor::from_vec_f64(vec![f64::NAN,-0.,0.,f64::NAN,2.,-2.],&[6]).unwrap();
        let (v,i)=x.sort(0,descending,true).unwrap();
        let expected=if descending {vec![0,3,4,1,2,5]} else {vec![5,1,2,4,0,3]};
        assert_eq!(i.to_vec_i64(),expected);
        assert_eq!(v.to_vec_f64()[if descending {3}else{1}].to_bits(),(-0.0f64).to_bits());
    }
}

#[test]
fn strided_scalar_empty_and_capacity_paths_are_checked_before_work() {
    let x=Tensor::from_vec(vec![3.,1.,2.,0.,5.,4.],&[2,3]).unwrap().transpose(0,1).unwrap();
    assert_eq!(x.sort(0,false,true).unwrap().0.to_vec(),vec![1.,0.,2.,4.,3.,5.]);
    let x=Tensor::scalar(3.).requires_grad_(true).unwrap();
    let (v,i)=x.sort(0,false,true).unwrap();
    assert_eq!(v.shape(),&[]);assert_eq!(i.to_vec_i64(),vec![0]);
    v.backward();assert_eq!(x.grad().unwrap().to_vec(),vec![1.]);
    for shape in [vec![1usize<<60,0],vec![0,1usize<<60],vec![0,3,2]] {
        let x=Tensor::from_vec(Vec::new(),&shape).unwrap().requires_grad_(true).unwrap();
        for dim in 0..shape.len() {
            let (v,i)=x.sort(dim,false,true).unwrap();
            assert_eq!(v.shape(),shape);assert_eq!(i.numel(),0);
            v.sum().backward();assert_eq!(x.grad().unwrap().numel(),0);x.zero_grad();
        }
    }
    let x=Tensor::scalar(1.).expand(&[1usize<<61]).unwrap();
    assert!(x.sort(0,false,true).is_err());
    assert!(Tensor::scalar(1.).sort(1,false,true).is_err());
    assert_eq!(x.device(),Device::Cpu);
}

#[test]
fn float_permutation_gradients_match_finite_difference() {
    let x=Tensor::from_vec(vec![0.9,-0.4,1.6,0.1,-1.2,0.7],&[2,3]).unwrap();
    grad_check(&[x],|t|t[0].sort(1,false,true).unwrap().0.square().unwrap().sum());
    let x=Tensor::from_vec_f64(vec![3.,1.,2.],&[3]).unwrap().requires_grad_(true).unwrap();
    let (v,_) = x.sort(0,false,true).unwrap();
    v.mul(&Tensor::from_vec_f64(vec![2.,4.,7.],&[3]).unwrap()).unwrap().sum().backward();
    assert_eq!(x.grad().unwrap().to_vec_f64(),vec![7.,2.,4.]);
}

#[test]
fn sort_saves_indices_without_saving_input_or_value_versions() {
    let x=Tensor::from_vec(vec![3.,1.,2.],&[3]).unwrap().requires_grad_(true).unwrap();
    let (v,_) = x.sort(0,false,true).unwrap();
    x.detach().fill_(8.).unwrap();v.detach().fill_(9.).unwrap();
    v.sum().backward();assert_eq!(x.grad().unwrap().to_vec(),vec![1.,1.,1.]);
    x.zero_grad();
    let (v,i)=x.sort(0,false,true).unwrap();
    i.zero_().unwrap();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(||v.sum().backward())).is_err());
    assert!(x.grad().is_none());
    assert!(ferro_core::capture(||x.sort(0,false,true)).is_err());
}

#[test]
fn sort_backward_preserves_signed_zero_cotangents() {
    for shape in [vec![],vec![1],vec![2]] {
        let size = shape.iter().product();
        let x=Tensor::from_vec(vec![1.;size],&shape).unwrap().requires_grad_(true).unwrap();
        let (v,_) = x.sort(0,false,true).unwrap();
        v.backward_with(&Tensor::from_vec(vec![-0.;size],&shape).unwrap());
        assert!(x.grad().unwrap().to_vec().iter().all(|x|x.to_bits()==(-0.0f32).to_bits()));
        let x=Tensor::from_vec_f64(vec![1.;size],&shape).unwrap().requires_grad_(true).unwrap();
        let (v,_) = x.sort(0,false,true).unwrap();
        v.backward_with(&Tensor::from_vec_f64(vec![-0.;size],&shape).unwrap());
        assert!(x.grad().unwrap().to_vec_f64().iter().all(|x|x.to_bits()==(-0.0f64).to_bits()));
    }
}
