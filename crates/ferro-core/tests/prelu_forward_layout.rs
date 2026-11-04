use ferro_core::{DType, Tensor};
use ferro_core::autograd::with_grad_enabled;
use ferro_core::forward_ad::DualLevel;

fn dimensions(value: &str) -> Vec<usize> {
    if value.is_empty() { Vec::new() } else { value.split(',').map(|x| x.parse().unwrap()).collect() }
}

fn check(actual: Option<&Tensor>, expected: &str, context: &str) {
    if expected == "-" { assert!(actual.is_none(), "{context}: expected absent"); return; }
    let value = actual.unwrap_or_else(|| panic!("{context}: unexpectedly absent"));
    let parts = expected.split('|').collect::<Vec<_>>();
    assert_eq!(value.dtype(), if parts[0] == "float64" { DType::F64 } else { DType::F32 }, "{context}: dtype");
    assert_eq!(value.shape(), dimensions(parts[1]), "{context}: shape");
    assert_eq!(value.strides(), dimensions(parts[2]), "{context}: stride");
    assert_eq!(value.requires_grad(), parts[3] == "1", "{context}: flag");
    assert_eq!(value.is_structural_zero(), parts[4] == "1", "{context}: marker");
    let expected = if parts[5].is_empty() { Vec::new() } else { parts[5].split(',').map(|x| x.parse::<f64>().unwrap()).collect::<Vec<_>>() };
    for (index, (a,b)) in value.to_vec_f64().iter().zip(&expected).enumerate() {
        assert!(a.is_nan() && b.is_nan() || a == b || (a.is_finite() && b.is_finite() && (a-b).abs() <= 2e-7 + 2e-7*b.abs()), "{context} value {index}: {a} != {b}");
        if *a == 0. && *b == 0. { assert_eq!(a.is_sign_negative(), b.is_sign_negative(), "{context} signed zero {index}"); }
    }
}

#[test]
fn logical_axis_arithmetic_preserves_all_54_torch_layout_flag_and_zero_policies() {
    let mut count = 0;
    for line in include_str!("fixtures/prelu_layout_reference.tsv").lines().filter(|line| !line.starts_with('#')) {
        let fields = line.split('\t').collect::<Vec<_>>();
        let xt = fields[1] == "1"; let vt = fields[2] == "1";
        let mode: usize = fields[3].parse().unwrap(); let wide = fields[4] == "1"; let enabled = fields[5] == "1";
        let nonfinite = fields[0].parse::<usize>().unwrap() >= 48;
        let shape = if nonfinite { [7,2,2] } else { [3,2,2] };
        let reversed = [shape[2],shape[1],shape[0]];
        let values = if nonfinite { vec![f32::NEG_INFINITY,-2.,-0.,0.,2.,f32::INFINITY,f32::NAN].repeat(4) }
                     else { vec![-2.,0.,3.,1.,-3.,2.,4.,-1.,-0.,-2.,1.,3.] };
        let x = Tensor::from_vec(values, if xt { &reversed } else { &shape }).unwrap();
        let x = if xt { x.transpose(0,2).unwrap() } else { x }.requires_grad_(true).unwrap();
        let w = Tensor::from_vec(if nonfinite { vec![f32::INFINITY,0.3] } else { vec![0.2,0.7] }, &[2]).unwrap().requires_grad_(true).unwrap();
        let dshape = if vt { &reversed } else { &shape };
        let dx = if wide { Tensor::from_vec_f64(if nonfinite { vec![0.7;28] } else { vec![0.3,-0.7,0.,1.1].repeat(3) }, dshape).unwrap() }
                 else { Tensor::from_vec(if nonfinite { vec![0.7;28] } else { vec![0.3,-0.7,0.,1.1].repeat(3) }, dshape).unwrap() };
        let dx = if vt { dx.transpose(0,2).unwrap() } else { dx }.requires_grad_(true).unwrap();
        let dw = if wide { Tensor::from_vec_f64(if nonfinite { vec![0.,0.2] } else { vec![0.4,-0.2] }, &[2]).unwrap() }
                 else { Tensor::from_vec(if nonfinite { vec![0.,0.2] } else { vec![0.4,-0.2] }, &[2]).unwrap() }.requires_grad_(true).unwrap();
        let (primals, directions) = match mode { 0 => (vec![&x],vec![&dx]), 1 => (vec![&w],vec![&dw]), _ => (vec![&x,&w],vec![&dx,&dw]) };
        let level = DualLevel::new(&primals,&directions).unwrap();
        let output = with_grad_enabled(enabled, || x.prelu(&w).unwrap());
        let tangent = level.tangent(&output).unwrap().unwrap(); drop(level);
        for result in [&output,&tangent] { for input in [&x,&w,&dx,&dw] { assert_ne!(result._storage_ptr(), input._storage_ptr()); } }
        check(Some(&output), fields[6], &format!("case {} primal",fields[0]));
        check(Some(&tangent), fields[7], &format!("case {} tangent",fields[0]));
        let gradients = if enabled {
            let seed = if tangent.dtype() == DType::F64 { Tensor::from_vec_f64(vec![1.;x.numel()], &shape).unwrap() } else { Tensor::ones(&shape) };
            Tensor::vjp_many_wrt_optional(&[&tangent], &[&x,&w,&dx,&dw], &[&seed], !wide).unwrap_or_else(|error| panic!("case {}: {error:?}", fields[0]))
        } else { vec![None,None,None,None] };
        for (index, gradient) in gradients.iter().enumerate() { check(gradient.as_ref(),fields[8+index],&format!("case {} gradient {index}",fields[0])); }
        for input in [&x,&w,&dx,&dw] { assert!(input.grad().is_none()); }
        count += 1;
    }
    assert_eq!(count,54);
}
