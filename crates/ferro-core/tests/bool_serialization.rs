use ferro_core::safetensors::{from_safetensors_bytes, to_safetensors_bytes};
use ferro_core::{DType, Error, Tensor};

fn encoded(header: &str, data: &[u8]) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    bytes
}

#[test]
fn bool_safetensors_roundtrip_scalar_empty_and_strided() {
    let base = Tensor::from_vec_bool(vec![true, false, false, true, true, false], &[2, 3]).unwrap();
    let tensors = [base.transpose(0, 1).unwrap(), Tensor::from_vec_bool(vec![true], &[]).unwrap(),
        Tensor::from_vec_bool(Vec::new(), &[2, 0, 3]).unwrap()];
    for tensor in tensors {
        let bytes = to_safetensors_bytes(&[("mask", &tensor)]).unwrap();
        let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert!(std::str::from_utf8(&bytes[8..8 + hlen]).unwrap().contains("\"dtype\":\"BOOL\""));
        assert_eq!(&bytes[8 + hlen..], tensor.to_vec_bool().into_iter().map(u8::from).collect::<Vec<_>>());
        let loaded = from_safetensors_bytes(&bytes).unwrap().remove(0).1;
        assert_eq!(loaded.dtype(), DType::Bool);
        assert_eq!(loaded.shape(), tensor.shape());
        assert_eq!(loaded.to_vec_bool(), tensor.to_vec_bool());
        assert!(!loaded.requires_grad());
        assert_eq!(to_safetensors_bytes(&[("mask", &loaded)]).unwrap(), bytes);
    }
}

#[test]
fn bool_foreign_bytes_are_normalized_without_invalid_rust_bool_reads() {
    let bytes = encoded(r#"{"mask":{"dtype":"BOOL","shape":[4],"data_offsets":[0,4]}}"#, &[0, 1, 2, 255]);
    let tensor = from_safetensors_bytes(&bytes).unwrap().remove(0).1;
    assert_eq!(tensor.to_vec_bool(), vec![false, true, true, true]);
    let canonical = to_safetensors_bytes(&[("mask", &tensor)]).unwrap();
    assert_eq!(&canonical[canonical.len() - 4..], &[0, 1, 1, 1]);
}

#[test]
fn malformed_bool_shapes_and_lengths_fail_without_panicking() {
    let huge = usize::MAX;
    let cases = [
        u64::MAX.to_le_bytes().to_vec(),
        encoded(r#"{"mask":{"dtype":"BOOL","shape":[2],"data_offsets":[0,1]}}"#, &[1]),
        encoded(r#"{"mask":{"dtype":"BOOL","shape":[0],"data_offsets":[0,1]}}"#, &[1]),
        encoded(&format!(r#"{{"mask":{{"dtype":"BOOL","shape":[0,{huge},2],"data_offsets":[0,0]}}}}"#), &[]),
        encoded(&format!(r#"{{"mask":{{"dtype":"BOOL","shape":[{huge},2,0],"data_offsets":[0,0]}}}}"#), &[]),
        encoded(&format!(r#"{{"mask":{{"dtype":"F64","shape":[{huge}],"data_offsets":[0,0]}}}}"#), &[]),
    ];
    for bytes in cases {
        assert!(matches!(from_safetensors_bytes(&bytes), Err(Error::Format { .. })));
    }
}
