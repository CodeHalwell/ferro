use ferro_core::{capture, DType, Tensor};

fn double(values: &[f64], shape: &[usize]) -> Tensor { Tensor::from_vec_f64(values.to_vec(), shape).unwrap() }
fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) { assert!((a - b).abs() <= tolerance * (1.0 + b.abs()), "{a} != {b}"); }
}

#[test]
fn double_unaries_have_finite_difference_adjoints() {
    let operations: &[fn(&Tensor) -> Tensor] = &[
        |x| x.sin().unwrap(), |x| x.cos().unwrap(), |x| x.tan().unwrap(),
        |x| x.sinh().unwrap(), |x| x.cosh().unwrap(), |x| x.exp2().unwrap(),
        |x| x.expm1().unwrap(), |x| x.log1p().unwrap(), |x| x.log2().unwrap(),
        |x| x.log10().unwrap(), |x| x.reciprocal().unwrap(), |x| x.silu(), |x| x.gelu(),
    ];
    let values = [0.15, 0.35, 0.75, 1.1];
    let weights = [0.3, -0.5, 0.7, 1.2];
    for operation in operations {
        let x = double(&values, &[2, 2]).requires_grad_(true).unwrap();
        let output = operation(&x);
        assert_eq!(output.dtype(), DType::F64);
        output.mul(&double(&weights, &[2, 2])).unwrap().sum().backward();
        let mut numeric = Vec::new();
        for index in 0..values.len() {
            let mut plus = values; let mut minus = values;
            plus[index] += 1e-6; minus[index] -= 1e-6;
            let high = operation(&double(&plus, &[2, 2])).to_vec_f64();
            let low = operation(&double(&minus, &[2, 2])).to_vec_f64();
            numeric.push((high[index] - low[index]) * weights[index] / 2e-6);
        }
        close(&x.grad().unwrap().to_vec_f64(), &numeric, 2e-9);
        assert_eq!(operation(&double(&[], &[2, 0])).shape(), &[2, 0]);
    }
}

#[test]
fn softmax_both_modes_have_strided_axis_adjoints_and_stable_offsets() {
    for logarithmic in [false, true] { for dim in 0..3 {
        let x = double(&(0..24).map(|i| i as f64 / 7.0 - 1.3).collect::<Vec<_>>(), &[2, 3, 4])
            .requires_grad_(true).unwrap();
        let transposed = x.transpose(0, 2).unwrap();
        let apply = |t: &Tensor| if logarithmic { t.log_softmax(dim).unwrap() } else { t.softmax(dim).unwrap() };
        let output = apply(&transposed);
        let weights = (0..24).map(|i| (i as f64 * 0.7).sin()).collect::<Vec<_>>();
        output.mul(&double(&weights, output.shape())).unwrap().sum().backward();
        let actual = transposed.grad().unwrap().to_vec_f64();
        let values = transposed.to_vec_f64();
        for index in 0..24 {
            let mut plus = values.clone(); let mut minus = values.clone();
            plus[index] += 1e-6; minus[index] -= 1e-6;
            let loss = |data: &[f64]| apply(&double(data, transposed.shape())).to_vec_f64()
                .iter().zip(&weights).map(|(v, w)| v * w).sum::<f64>();
            close(&[actual[index]], &[(loss(&plus) - loss(&minus)) / 2e-6], 5e-9);
        }
    } }
    let large = double(&[1e100; 3], &[3]);
    close(&large.log_softmax(0).unwrap().to_vec_f64(), &[-3f64.ln(); 3], 1e-15);
    assert_eq!(large.softmax(0).unwrap().to_vec_f64(), vec![1.0 / 3.0; 3]);
}

#[test]
fn double_softmax_empty_scalar_capacity_and_capture_contracts() {
    for shape in [vec![2, 0, 3], vec![0, 2, 3], vec![2, 3, 0], vec![1 << 60, 0]] {
        let x = double(&[], &shape).requires_grad_(true).unwrap();
        for dim in 0..shape.len() {
            let out = x.softmax(dim).unwrap().add(&x.log_softmax(dim).unwrap()).unwrap();
            assert_eq!(out.shape(), shape);
            out.sum().backward();
            assert!(x.grad().unwrap().to_vec_f64().is_empty());
        }
    }
    let scalar = double(&[2.0], &[]).requires_grad_(true).unwrap();
    assert_eq!(scalar.softmax(0).unwrap().to_vec_f64(), vec![1.0]);
    scalar.log_softmax(0).unwrap().backward();
    assert_eq!(scalar.grad().unwrap().to_vec_f64(), vec![0.0]);
    assert!(scalar.softmax(1).is_err());
    assert!(capture(|| scalar.softmax(0)).is_err());
    assert!(capture(|| scalar.sin()).is_err());
    let huge = scalar.expand(&[isize::MAX as usize / 8 + 1]).unwrap();
    assert!(huge.softmax(0).is_err());
    assert!(huge.log_softmax(0).is_err());
    assert!(huge.sin().is_err());
}

#[test]
fn near_zero_transcendentals_preserve_precision_and_reciprocal_matches_ieee_order() {
    let small = double(&[1e-300, -1e-300], &[2]);
    assert_eq!(small.expm1().unwrap().to_vec_f64(), vec![1e-300, -1e-300]);
    assert_eq!(small.log1p().unwrap().to_vec_f64(), vec![1e-300, -1e-300]);
    let x = double(&[1e-200, 1e200], &[2]).requires_grad_(true).unwrap();
    let seed = double(&[1e-200, 1e200], &[2]);
    let derivative = x.reciprocal().unwrap().vjp_wrt(&[&x], &seed, false).unwrap().remove(0);
    assert_eq!(derivative.to_vec_f64(), vec![f64::NEG_INFINITY, -0.0]);
}
