use ferro_core::{Tensor, capture};
use ferro_core::ops_ext::embedding_bag::embedding_bag;
use ferro_core::autograd::with_grad_enabled;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountAlloc;
thread_local! {
    static WATCH: Cell<bool> = const { Cell::new(false) };
    static TOTAL: Cell<usize> = const { Cell::new(0) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}
#[global_allocator]
static ALLOC: CountAlloc = CountAlloc;
fn allocation(bytes: usize) {
    let _ = WATCH.try_with(|watch| { if watch.get() {
        let _ = TOTAL.try_with(|total| total.set(total.get().saturating_add(bytes)));
        let _ = LARGEST.try_with(|largest| largest.set(largest.get().max(bytes)));
    }});
}
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 { allocation(layout.size()); unsafe { System.alloc(layout) } }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 { allocation(layout.size()); unsafe { System.alloc_zeroed(layout) } }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 { allocation(size); unsafe { System.realloc(pointer,layout,size) } }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) { unsafe { System.dealloc(pointer,layout) } }
}

fn ids(values: &[i64]) -> Tensor { Tensor::from_vec_i64(values.to_vec(), &[values.len()]).unwrap() }
fn bag(weight: &Tensor, input: &Tensor, offsets: &Tensor, mode: &str) -> Tensor {
    embedding_bag(weight, input, Some(offsets), mode, None, false, None, None, 2., false).unwrap()
}

#[test]
fn all_modes_and_live_sample_weights_pass_finite_differences() {
    let weight = Tensor::from_vec(vec![0.2, -0.4, 0.7, 1.1, -0.3, 0.8, 1.3, -0.9], &[4,2]).unwrap();
    let input = ids(&[2,0,2,1,3,1]); let offsets = ids(&[0,3,3,5]);
    for mode in ["sum", "mean", "max"] {
        ferro_core::testkit::grad_check(&[weight.clone()], |v| bag(&v[0], &input, &offsets, mode).square().unwrap().sum());
    }
    let samples = Tensor::from_vec(vec![0.3, -0.2, 0.5, 1.1, 0.7, -0.6], &[6]).unwrap();
    ferro_core::testkit::grad_check(&[weight,samples], |v| embedding_bag(&v[0], &input, Some(&offsets), "sum", Some(&v[1]), false, None, None, 2., false).unwrap().square().unwrap().sum());
}

#[test]
fn typed_padding_empty_bags_and_current_torch_frequency_rule() {
    let weight = Tensor::from_vec_f64(vec![1.,2.,3.,4.,5.,6.], &[3,2]).unwrap().requires_grad_(true).unwrap();
    let input = ids(&[0,0,0,1,2,2]); let offsets = ids(&[0,3,3]);
    let output = embedding_bag(&weight,&input,Some(&offsets),"mean",None,false,Some(0),None,2.,true).unwrap();
    assert_eq!(output.to_vec_f64(), vec![0.,0.,0.,0.,13./3.,16./3.]);
    output.sum().backward();
    assert_eq!(weight.grad().unwrap().to_vec_f64(), vec![0.,0.,1./9.,1./9.,2./9.,2./9.]);
}

#[test]
fn max_first_ties_nan_and_padding_are_selected_without_zero_sentinels() {
    let weight = Tensor::from_vec(vec![f32::NAN, 4., 2., f32::NAN, 3.,5., 3.,5.], &[4,2]).unwrap().requires_grad_(true).unwrap();
    let input = ids(&[0,1,2,3]); let offsets = ids(&[0]);
    let output = bag(&weight, &input, &offsets, "max");
    assert!(output.to_vec()[0].is_nan()); assert_eq!(output.to_vec()[1],5.);
    output.sum().backward();
    assert_eq!(weight.grad().unwrap().to_vec(),vec![1.,0.,0.,0.,0.,1.,0.,0.]);
}

#[test]
fn saved_metadata_and_sample_values_check_versions_but_plain_weight_does_not() {
    let weight = Tensor::ones(&[3,2]).requires_grad_(true).unwrap();
    let input = ids(&[0,1,2]); let offsets = ids(&[0,2]);
    let output = bag(&weight,&input,&offsets,"sum");
    with_grad_enabled(false, || weight.copy_leaf_from(&Tensor::zeros(&[3,2])).unwrap());
    output.sum().backward();
    for which in 0..3 {
        let samples = Tensor::ones(&[3]).requires_grad_(true).unwrap();
        let output = embedding_bag(&weight,&input,Some(&offsets),"sum",Some(&samples),false,None,None,2.,false).unwrap();
        if which == 0 { input.copy_from(&ids(&[1,1,2])).unwrap(); }
        else if which == 1 { offsets.copy_from(&ids(&[0,1])).unwrap(); }
        else { with_grad_enabled(false, || samples.copy_leaf_from(&Tensor::zeros(&[3])).unwrap()); }
        assert!(output.vjp_wrt(&[&weight], &Tensor::ones(&[2,2]), false).is_err());
    }
    let output = bag(&weight,&input,&offsets,"sum");
    assert!(output.grad_wrt(&[&weight],true).is_err());
}

#[test]
fn malformed_metadata_precedes_renorm_and_empty_width_remains_bounded() {
    let weight = Tensor::from_vec(vec![3.,4.,0.,2.], &[2,2]).unwrap();
    let before=weight.to_vec(); let version=weight._version();
    for offsets in [ids(&[]),ids(&[1]),ids(&[0,2,1]),ids(&[0,-1]),ids(&[0,4])] {
        assert!(embedding_bag(&weight,&ids(&[0,1,0]),Some(&offsets),"sum",None,false,None,Some(1.),2.,false).is_err());
        assert_eq!(weight.to_vec(),before); assert_eq!(weight._version(),version);
    }
    assert!(embedding_bag(&weight,&ids(&[0,1]),Some(&ids(&[0,1])),"sum",None,true,None,Some(1.),2.,false).is_err());
    let count=1_000_000_000_000usize;
    let empty=Tensor::from_vec_f64(vec![],&[count,0]).unwrap().requires_grad_(true).unwrap();
    let input=ids(&[1]).reshape(&[1,1]).unwrap().expand(&[count,1]).unwrap();
    let output=embedding_bag(&empty,&input,None,"max",None,false,None,Some(1.),2.,false).unwrap();
    assert_eq!(output.shape(),&[count,0]);output.sum().backward();assert_eq!(empty.grad().unwrap().shape(),&[count,0]);
    let samples=Tensor::from_vec_f64(vec![2.],&[1,1]).unwrap().expand(&[count,1]).unwrap();
    let output=embedding_bag(&empty,&input,None,"sum",Some(&samples),false,None,None,2.,false).unwrap();
    assert_eq!(output.shape(),&[count,0]);output.sum().backward();assert_eq!(empty.grad().unwrap().shape(),&[count,0]);
    assert!(capture(|| embedding_bag(&weight,&ids(&[0]),Some(&ids(&[0])),"sum",None,false,None,None,2.,false)).is_err());
}

#[test]
fn forward_does_not_allocate_a_token_by_width_embedding_matrix() {
    let weight=Tensor::ones(&[4,512]);
    let input=ids(&(0..8192).map(|i| i%4).collect::<Vec<_>>()); let offsets=ids(&[0,2048,4096,6144]);
    for mode in ["sum","mean","max"] {
        TOTAL.set(0); LARGEST.set(0); WATCH.set(true);
        let output=bag(&weight,&input,&offsets,mode);
        WATCH.set(false);
        assert!(LARGEST.get()<256*1024,"largest allocation {}",LARGEST.get());
        assert!(TOTAL.get()<1024*1024,"total allocated {}",TOTAL.get());
        assert_eq!(output.shape(),&[4,512]);
        assert_eq!(output.to_vec()[0],if mode=="sum" {2048.} else {1.});
    }
}
