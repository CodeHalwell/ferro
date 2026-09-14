use ferro_core::data::{DataLoader, TensorDataset};
use ferro_core::nn::cross_entropy_indices;
use ferro_core::{DType, Tensor};
use std::sync::Arc;

#[test]
fn integer_dataset_to_loss_train_step() {
    let features = Tensor::from_vec(vec![1., 0., 0., 1., 1., 1.], &[3, 2]).unwrap();
    let labels = Tensor::from_vec_i64(vec![1, 0, 1], &[3]).unwrap();
    for workers in [0, 2] {
        let ds = Arc::new(TensorDataset::new(features.clone(), labels.clone()).unwrap());
        let loader = DataLoader::new(ds, 2).workers(workers);
        let mut weight = Tensor::from_vec(vec![0.1, -0.2, 0.3, 0.4], &[2, 2])
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let mut reference = weight.detach_copy().requires_grad_(true).unwrap();
        let mut seen = 0;
        for batch in loader.iter() {
            let (x, ids) = batch.unwrap();
            let n = x.shape()[0];
            let expected = &[1i64, 0, 1][seen..seen + n];
            assert_eq!(ids.dtype(), DType::I64);
            assert_eq!(ids.shape(), &[n]);
            assert_eq!(ids.to_vec_i64(), expected);
            let logits = x.matmul(&weight).unwrap();
            let loss = cross_entropy_indices(&logits, &ids).unwrap();
            let expected_ids = Tensor::from_vec_i64(expected.to_vec(), &[n]).unwrap();
            let ref_loss =
                cross_entropy_indices(&x.matmul(&reference).unwrap(), &expected_ids).unwrap();
            assert_eq!(loss.to_vec(), ref_loss.to_vec());
            loss.backward();
            ref_loss.backward();
            let grad = weight.grad().unwrap().to_vec();
            assert_eq!(grad, reference.grad().unwrap().to_vec());
            assert!(grad.iter().all(|g| g.is_finite()));
            assert!(grad.iter().any(|&g| g != 0.));
            let before = weight.to_vec();
            let updated: Vec<f32> = before.iter().zip(&grad).map(|(w, g)| w - 0.1 * g).collect();
            let ref_updated: Vec<f32> = reference
                .to_vec()
                .iter()
                .zip(reference.grad().unwrap().to_vec())
                .map(|(w, g)| w - 0.1 * g)
                .collect();
            assert_ne!(updated, before);
            assert_eq!(updated, ref_updated);
            weight = Tensor::from_vec(updated, &[2, 2])
                .unwrap()
                .requires_grad_(true)
                .unwrap();
            reference = Tensor::from_vec(ref_updated, &[2, 2])
                .unwrap()
                .requires_grad_(true)
                .unwrap();
            seen += n;
        }
        assert_eq!(seen, 3);
    }
}

#[test]
fn typed_cat_preserves_strided_i64_and_rejects_mixed_dtype() {
    let x = Tensor::from_vec_i64(vec![i64::MAX, i64::MIN, 16_777_217, -16_777_217], &[2, 2])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let empty = Tensor::from_vec_i64(vec![], &[2, 0]).unwrap();
    let y = Tensor::cat(&[x.clone(), empty, x.clone()], 1).unwrap();
    assert_eq!(y.dtype(), DType::I64);
    assert_eq!(
        y.to_vec_i64(),
        vec![
            i64::MAX,
            16_777_217,
            i64::MAX,
            16_777_217,
            i64::MIN,
            -16_777_217,
            i64::MIN,
            -16_777_217
        ]
    );
    assert!(Tensor::cat(&[x.clone(), x.to_dtype(DType::F32)], 0).is_err());
    assert!(Tensor::cat(&[x.clone()], 2).is_err());
    assert!(Tensor::cat(&[], 0).is_err());
}

#[test]
fn cat_axis_overflow_is_an_error_for_empty_tensors() {
    let x = Tensor::from_vec_i64(vec![], &[usize::MAX, 0]).unwrap();
    assert!(Tensor::cat(&[x.clone(), x], 0).is_err());
}

#[test]
fn typed_float_and_half_data_are_bit_exact() {
    let bits = [
        1.0000000000000002f64.to_bits(),
        (-0.0f64).to_bits(),
        0x7ff8000000000042,
        f64::MAX.to_bits(),
    ];
    let x = Tensor::from_vec_f64(bits.iter().map(|&b| f64::from_bits(b)).collect(), &[2, 2])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let y = x.index_select(0, &[1, 0]).unwrap();
    assert_eq!(y.dtype(), DType::F64);
    assert_eq!(
        y.to_vec_f64()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        vec![bits[1], bits[3], bits[0], bits[2]]
    );
    let z = Tensor::cat(&[y.clone(), y], 0).unwrap();
    assert_eq!(z.dtype(), DType::F64);
    assert_eq!(
        z.to_vec_f64()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        vec![bits[1], bits[3], bits[0], bits[2], bits[1], bits[3], bits[0], bits[2]]
    );
    for dtype in [DType::F16, DType::BF16] {
        let raw = vec![0x8000, 0x0001, 0x7c01, 0xffff];
        let x = match dtype {
            DType::F16 => Tensor::from_vec_f16_bits(raw, &[2, 2]),
            _ => Tensor::from_vec_bf16_bits(raw, &[2, 2]),
        }
        .unwrap()
        .transpose(0, 1)
        .unwrap();
        let y = x.index_select(1, &[1, 0]).unwrap();
        let z = Tensor::cat(&[y.clone(), y], 1).unwrap();
        assert_eq!(z.dtype(), dtype);
        let bits = if dtype == DType::F16 {
            z.to_vec_f16_bits()
        } else {
            z.to_vec_bf16_bits()
        }
        .unwrap();
        assert_eq!(
            bits,
            vec![0x7c01, 0x8000, 0x7c01, 0x8000, 0xffff, 0x0001, 0xffff, 0x0001]
        );
        assert_eq!(x.index_select(0, &[]).unwrap().dtype(), dtype);
    }
}

#[test]
fn empty_and_large_integer_datasets_keep_exact_labels() {
    use ferro_core::data::{CollateFn, Dataset, StackCollate};
    let labels = vec![i64::MAX, 9_007_199_254_740_993, i64::MIN];
    let ds = Arc::new(
        TensorDataset::new(
            Tensor::from_vec(vec![1., 2., 3.], &[3, 1]).unwrap(),
            Tensor::from_vec_i64(labels.clone(), &[3]).unwrap(),
        )
        .unwrap(),
    );
    assert_eq!(ds.get(1).unwrap().1.to_vec_i64(), vec![labels[1]]);
    assert!(ds.get(3).is_err());
    let loader = DataLoader::new(ds.clone(), 2).workers(2);
    let got: Vec<i64> = loader
        .iter()
        .flat_map(|b| b.unwrap().1.to_vec_i64())
        .collect();
    assert_eq!(got, labels);
    assert_eq!(DataLoader::new(ds, 2).drop_last(true).iter().count(), 1);
    let empty = Arc::new(
        TensorDataset::new(
            Tensor::from_vec(vec![], &[0, 1]).unwrap(),
            Tensor::from_vec_i64(vec![], &[0]).unwrap(),
        )
        .unwrap(),
    );
    assert_eq!(DataLoader::new(empty, 2).iter().count(), 0);
    assert!(StackCollate.collate(&[]).is_err());
}

#[test]
fn f32_duplicate_selection_and_cat_accumulate_gradients() {
    let x = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let ids = Tensor::from_vec_i64(vec![1, 1], &[2]).unwrap();
    let y = x.index_select_t(0, &ids).unwrap();
    Tensor::cat(&[y.clone(), y], 1).unwrap().sum().backward();
    assert_eq!(x.grad().unwrap().to_vec(), vec![0., 0., 4., 4.]);
}

#[test]
fn typed_index_select_preserves_exact_strided_values() {
    let values = vec![
        i64::MAX,
        16_777_217,
        i64::MIN,
        -16_777_217,
        9_007_199_254_740_993,
        7,
    ];
    let x = Tensor::from_vec_i64(values.clone(), &[2, 3])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let y = x.index_select(1, &[1, 0, 1]).unwrap();
    assert_eq!(y.dtype(), DType::I64);
    assert_eq!(
        y.to_vec_i64(),
        vec![
            values[3], values[0], values[3], values[4], values[1], values[4], values[5], values[2],
            values[5]
        ]
    );
    let ids = Tensor::from_vec_i64(vec![1, 0, 1], &[3]).unwrap();
    assert_eq!(
        x.index_select_t(1, &ids).unwrap().to_vec_i64(),
        y.to_vec_i64()
    );
    assert_eq!(x.index_select(0, &[]).unwrap().dtype(), DType::I64);
    assert_eq!(x.index_select(0, &[]).unwrap().shape(), &[0, 2]);
    assert!(x.index_select(2, &[0]).is_err());
    assert!(x.index_select(0, &[3]).is_err());
}
