use ferro_core::{Tensor, DType, Result, Device};
use ferro_core::autograd::{with_grad_enabled, is_grad_enabled};
use ferro_core::forward_ad::{DualLevel, is_active};

struct Golden {
    input: &'static [usize], data: &'static [f32], enabled: bool, live: bool, kind: &'static str,
    shape: &'static [usize], strides: &'static [usize], values: &'static [f64], requires: bool,
    wide: bool, gradient: Option<&'static [f64]>, marker: bool,
}
include!("fixtures/physical_forward_goldens.rs");

fn evaluate(golden: &Golden) -> Result<(Tensor, Tensor)> {
    let x = Tensor::from_vec(golden.data.to_vec(), golden.input)?.requires_grad_(golden.live)?;
    let n = x.numel();
    let mut entries = vec![0.; n*n];
    for index in 0..n { entries[index*n+index] = 1.; }
    let mut basis_shape = vec![n]; basis_shape.extend_from_slice(golden.input);
    let basis = Tensor::from_vec(entries, &[n,n])?.view(&basis_shape)?;
    let jacobian = with_grad_enabled(golden.enabled, || -> Result<Tensor> {
        let level = DualLevel::functional_batched(&[&x], &[&basis], n)?;
        let y = match golden.kind {
            "identity" => x.clone(),
            "square" => x.mul(&x)?,
            "sign" => x.sign()?,
            "cast" => x.to_dtype_autograd(DType::F64, false)?,
            "wide_pow0" => x.to_dtype_autograd(DType::F64, false)?.powf64(0.)?,
            "constant" => Tensor::ones(&[2,3]),
            "sum" => x.mul(&x)?.sum(),
            "transpose" => x.transpose(0,1)?,
            "slice" => x.slice(1,0,x.shape()[1],2)?,
            "slice_square" => { let view = x.slice(1,0,x.shape()[1],2)?; view.mul(&view)? },
            "expand" => x.slice(1,0,1,1)?.expand(&[x.shape()[0],3])?,
            _ => unreachable!(),
        };
        let derivative = level.tangents(&y)?;
        let mut physical = vec![n]; physical.extend_from_slice(y.shape());
        let derivative = if let Some(mut values) = derivative { assert_eq!(values.len(),1); values.remove(0) }
            else { Tensor::full_on(y.shape(),0.,Device::Cpu)?.expand(&physical)? };
        drop(level);
        let mut axes = (1..=y.ndim()).collect::<Vec<_>>(); axes.push(0);
        let mut shape = y.shape().to_vec(); shape.extend_from_slice(x.shape());
        derivative.permute(&axes)?.view(&shape)
    })?;
    Ok((x,jacobian))
}

#[test]
fn actual_torch_layout_values_flags_and_reverse_gradients() {
    let mut errors = vec![];
    let mut unsupported = 0;
    for (index,golden) in GOLDENS.iter().enumerate() {
        let result = evaluate(golden);
        if golden.kind == "wide_pow0" {
            // The existing F64 pow is opaque, outside both direct and batched catalogs.
            assert!(matches!(result, Err(ferro_core::Error::Unsupported { .. })));
            unsupported += 1;
            continue;
        }
        let (x,jacobian) = match result { Ok(values) => values, Err(error) => { errors.push(format!("{index} {} {:?}: {error:?}",golden.kind,golden.input)); continue; } };
        if jacobian.shape()!=golden.shape || jacobian.strides()!=golden.strides || jacobian.requires_grad()!=golden.requires
            || jacobian.dtype()!=if golden.wide { DType::F64 } else { DType::F32 } {
            errors.push(format!("{index} {} {:?} live={} enabled={}: {:?}/{:?}/{} expected {:?}/{:?}/{}",golden.kind,golden.input,golden.live,golden.enabled,
                jacobian.shape(),jacobian.strides(),jacobian.requires_grad(),golden.shape,golden.strides,golden.requires));
        }
        for (actual,expected) in jacobian.to_vec_f64().iter().zip(golden.values) {
            if (actual-expected).abs()>1e-6 { errors.push(format!("{index}: value {actual} != {expected}")); break; }
        }
        if let Some(expected) = golden.gradient {
            let derivative = jacobian.sum().grad_wrt(&[&x],false).unwrap().remove(0);
            if derivative.is_structural_zero()!=golden.marker { errors.push(format!("{index}: marker {} != {}",derivative.is_structural_zero(),golden.marker)); }
            for (actual,expected) in derivative.to_vec_f64().iter().zip(expected) {
                if (actual-expected).abs()>1e-6 { errors.push(format!("{index}: gradient {actual} != {expected}")); break; }
            }
        }
        assert!(x.grad().is_none());
        assert!(is_grad_enabled() && !is_active());
    }
    assert!(errors.is_empty(), "{} errors across {} controls:\n{}",errors.len(),GOLDENS.len(),errors.join("\n"));
    println!("{} actual Torch layout/value/flag/gradient controls passed; {unsupported} inherited opaque F64 boundaries checked",GOLDENS.len()-unsupported);
}
