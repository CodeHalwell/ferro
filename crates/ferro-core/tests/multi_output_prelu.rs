use ferro_core::Tensor;

fn leaf(v: f32) -> Tensor { Tensor::scalar(v).requires_grad_(true).unwrap() }

#[test]
fn absent_tangent_zeros_preserve_copy_layout_without_reading_or_recording_values() {
    for wide in [false, true] {
        let base = if wide { Tensor::from_vec_f64(vec![f64::NAN; 12], &[3, 4]).unwrap() }
                   else { Tensor::from_vec(vec![f32::NAN; 12], &[3, 4]).unwrap() };
        let cases = [
            (base.transpose(0, 1).unwrap(), vec![1, 4]),
            (base.slice(1, 0, 4, 2).unwrap().transpose(0, 1).unwrap(), vec![1, 2]),
            (base.narrow(0, 0, 1).unwrap().expand(&[3, 4]).unwrap(), vec![4, 1]),
            (base.narrow(0, 0, 0).unwrap().transpose(0, 1).unwrap(), vec![1, 4]),
        ];
        for (value, stride) in cases {
            let value=value.requires_grad_(true).unwrap();
            let output=value.zeros_like_layout().unwrap();
            assert_eq!(output.shape(),value.shape());assert_eq!(output.strides(),stride);
            assert_eq!(output.dtype(),value.dtype());assert!(!output.requires_grad());
            if wide { assert!(output.to_vec_f64().iter().all(|v|v.to_bits()==0)); }
            else { assert!(output.to_vec().iter().all(|v|v.to_bits()==0)); }
            if output.numel()>0 {
                let infinity=if wide { Tensor::from_vec_f64(vec![f64::INFINITY],&[]).unwrap() } else { Tensor::scalar(f32::INFINITY) };
                let ordinary=output.mul(&infinity).unwrap();
                if wide { assert!(ordinary.to_vec_f64().iter().all(|v|v.is_nan())); }
                else { assert!(ordinary.to_vec().iter().all(|v|v.is_nan())); }
            }
        }
    }
    let empty=Tensor::zeros(&[0,1usize<<40,2]).transpose(0,2).unwrap();
    let output=empty.zeros_like_layout().unwrap();
    assert_eq!(output.shape(),empty.shape());assert_eq!(output.strides(),empty.strides());
    let huge=Tensor::scalar(1.).expand(&[isize::MAX as usize/4+1]).unwrap();
    assert!(huge.zeros_like_layout().is_err());
}

#[test]
fn public_prelu_joint_higher_query_preserves_absent_sibling_and_signed_zero() {
    for (xv,wv,a,b) in [(f32::NEG_INFINITY,f32::INFINITY,1.,0.),(2.,3.,-0.,0.)] {
        let x=leaf(xv);let w=leaf(wv);let g=leaf(1.);
        let y=x.prelu(&w).unwrap();
        let pair=Tensor::vjp_many_wrt_optional(&[&y],&[&x,&w],&[&g],true).unwrap();
        let gx=pair[0].as_ref().unwrap();let gw=pair[1].as_ref().unwrap();
        let a=Tensor::scalar(a);let b=Tensor::scalar(b);
        let single=gx.vjp_wrt_optional(&[&g],&a,true).unwrap().remove(0).unwrap();
        let both=Tensor::vjp_many_wrt_optional(&[gx,gw],&[&g],&[&a,&b],true).unwrap().remove(0).unwrap();
        if xv.is_infinite() { assert_eq!(single.item(),f32::INFINITY);assert!(both.item().is_nan()); }
        else { assert_eq!(single.item().to_bits(),(-0.0f32).to_bits());assert_eq!(both.item().to_bits(),(-0.0f32).to_bits()); }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn graph_query_prunes_unselected_huge_virtual_weight_reduction() {
    if std::env::var_os("FERRO_PRELU_EMPTY_CHILD").is_none() {
        let mut child=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","graph_query_prunes_unselected_huge_virtual_weight_reduction","--test-threads=1"])
            .env("FERRO_PRELU_EMPTY_CHILD","1").spawn().unwrap();
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(15);
        loop {
            if let Some(status)=child.try_wait().unwrap() { assert!(status.success()); return; }
            if std::time::Instant::now()>=deadline { child.kill().unwrap(); let _=child.wait(); panic!("empty PReLU query exceeded time bound"); }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    #[repr(C)] struct Limit { current: u64, maximum: u64 }
    unsafe extern "C" { fn setrlimit(resource: i32, limit: *const Limit) -> i32; }
    // A regression must fail allocation in its child, never touch terabytes.
    let limit=Limit { current: 2u64<<30, maximum: 2u64<<30 };
    assert_eq!(unsafe { setrlimit(9,&limit) },0);
    let size=1usize<<40;
    let x=Tensor::zeros(&[0,size,2]).requires_grad_(true).unwrap();
    let w=Tensor::scalar(0.3).expand(&[size]).unwrap();
    let y=x.prelu(&w).unwrap();
    let gradient=Tensor::vjp_many_wrt_optional(&[&y],&[&x],&[&Tensor::zeros(y.shape())],true).unwrap().remove(0).unwrap();
    assert_eq!(gradient.shape(),x.shape());assert_eq!(gradient.numel(),0);assert!(gradient.requires_grad());
    let second=gradient.vjp_wrt_optional(&[&x],&Tensor::zeros(x.shape()),true).unwrap().remove(0).unwrap();
    assert_eq!(second.shape(),x.shape());assert!(!second.requires_grad());
}

#[test]
fn public_first_gradients_keep_real_defined_zero_cross_derivatives() {
    for value in [-2.,-0.,0.,2.] {
        let x=leaf(value);let w=leaf(0.3);let g=leaf(1.2);
        let y=x.prelu(&w).unwrap();
        let pair=Tensor::vjp_many_wrt_optional(&[&y],&[&x,&w],&[&g],true).unwrap();
        let gx=pair[0].as_ref().unwrap();let gw=pair[1].as_ref().unwrap();
        let from_x=gx.vjp_wrt_optional(&[&x,&w,&g],&Tensor::scalar(1.),true).unwrap();
        let from_w=gw.vjp_wrt_optional(&[&x,&w,&g],&Tensor::scalar(1.),true).unwrap();
        assert!(from_x.iter().chain(&from_w).all(Option::is_some));
        assert_eq!(from_x[0].as_ref().unwrap().item(),0.);assert!(!from_x[0].as_ref().unwrap().requires_grad());
        assert_eq!(from_w[1].as_ref().unwrap().item(),0.);assert!(!from_w[1].as_ref().unwrap().requires_grad());
        assert_eq!(from_x[2].as_ref().unwrap().item(),if value>=0. {1.}else{0.3});
        assert_eq!(from_w[0].as_ref().unwrap().item(),if value>=0. {0.}else{1.2});
    }
}
