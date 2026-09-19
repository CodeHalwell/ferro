use ferro_core::{Device, Tensor};
use ferro_core::segment::PreparedSegments;
use ferro_core::sparse::Coo;
use ferro_core::testkit::grad_check_strict;
fn t(v: &[f32], s: &[usize]) -> Tensor { Tensor::from_vec(v.to_vec(), s).unwrap() }

#[test]
fn indexing_gradients_arbitrary_axes_and_duplicates() {
    let p = PreparedSegments::new(&[2,0,2],4,Device::Cpu).unwrap();
    let x = t(&[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], &[2,4]);
    assert_eq!(p.select(&x,1).unwrap().to_vec(), vec![0.3,0.1,0.3,0.7,0.5,0.7]);
    grad_check_strict(&[x.clone()], |xs| { let y=p.select(&xs[0],1).unwrap(); y.mul(&y).unwrap().sum() });
    let src=t(&[0.3,0.2,0.1,0.6,0.5,0.4], &[2,3]);
    grad_check_strict(&[x.clone(),src.clone()], |xs| { let y=p.scatter_add(&xs[0],&xs[1],1).unwrap(); y.mul(&y).unwrap().sum() });
    assert!(p.select(&x,2).is_err());
    assert!(p.select(&x,0).is_err());
    assert!(p.scatter_add(&x,&src,0).is_err());
    let empty=PreparedSegments::new(&[],4,Device::Cpu).unwrap();
    assert_eq!(empty.select(&x,1).unwrap().shape(), &[2,0]);
    assert_eq!(empty.scatter_add(&x,&t(&[], &[2,0]),1).unwrap().to_vec(),x.to_vec());
}

#[test]
fn sparse_gradients_and_shared_operand() {
    let a=Coo::new(3,2,vec![1,0,1],vec![0,1,0]).unwrap();
    let p=a.prepare(Device::Cpu).unwrap();
    let w=t(&[0.2,0.3,-0.1], &[3]);
    let x=t(&[0.1,0.2,0.4,0.5], &[2,2]);
    assert_eq!(p.spmm(&w,&x).unwrap().to_vec(),a.spmm(&w,&x).unwrap().to_vec());
    grad_check_strict(&[w,x], |xs| { let y=p.spmm(&xs[0],&xs[1]).unwrap(); y.mul(&y).unwrap().sum() });
    let l=t(&[0.1,0.2,0.3,0.4,0.5,0.6], &[3,2]);
    let r=t(&[0.2,-0.1,0.1,0.3], &[2,2]);
    grad_check_strict(&[l,r], |xs| { let y=p.sddmm(&xs[0],&xs[1]).unwrap(); y.mul(&y).unwrap().sum() });
    let p=Coo::new(2,2,vec![0,1,1],vec![1,0,0]).unwrap().prepare(Device::Cpu).unwrap();
    grad_check_strict(&[t(&[0.1,0.2,0.3,0.4], &[2,2])], |xs| p.sddmm(&xs[0],&xs[0]).unwrap().sum());
}

#[test]
fn batching_preserves_isolates_duplicates_and_rectangular_blocks() {
    let a=Coo::new(3,2,vec![1,0,1],vec![0,1,0]).unwrap();
    let b=Coo::new(2,3,vec![0,0],vec![1,2]).unwrap();
    let (batch, offsets)=Coo::disjoint_union(&[a,b]).unwrap();
    assert_eq!(offsets,vec![[0,0],[3,2],[5,5]]);
    assert_eq!(batch.rows(), &[1,0,1,3,3]);
    assert_eq!(batch.cols(), &[0,1,0,3,4]);
    let x=t(&[1.,2.,3.,4.,5.], &[5,1]);
    assert_eq!(batch.prepare(Device::Cpu).unwrap().spmm(&t(&[1.;5], &[5]),&x).unwrap().to_vec(),vec![2.,2.,0.,9.,0.]);
    assert_eq!(Coo::disjoint_union(&[]).unwrap().0.shape(),[0,0]);
    let huge=Coo::new(usize::MAX,0,vec![],vec![]).unwrap();
    assert!(Coo::disjoint_union(&[huge,Coo::new(1,0,vec![],vec![]).unwrap()]).is_err());
}
