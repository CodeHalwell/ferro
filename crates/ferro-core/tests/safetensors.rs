use ferro_core::safetensors::{
    from_safetensors_bytes, load_safetensors, save_safetensors, to_safetensors_bytes,
};
use ferro_core::{DType, Error, Tensor};

fn tmp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ferro_st_{}_{name}", std::process::id()));
    p
}

#[test]
fn roundtrip_all_dtypes_and_ranks() {
    let f = Tensor::from_vec(vec![1.5, -2.25, 0.0, 3.125, -0.5, 42.0], &[2, 3]).unwrap();
    let d = Tensor::from_vec_f64(vec![1e-300, -2.5, 3.75], &[3]).unwrap();
    let i = Tensor::from_vec_i64(vec![i64::MIN, -1, 0, i64::MAX], &[2, 2]).unwrap();
    let s = Tensor::scalar(7.5);
    let bytes =
        to_safetensors_bytes(&[("w", &f), ("prec", &d), ("ids", &i), ("step", &s)]).unwrap();

    let got = from_safetensors_bytes(&bytes).unwrap();
    assert_eq!(got.len(), 4);
    let (names, tensors): (Vec<_>, Vec<_>) = got.into_iter().unzip();
    assert_eq!(names, ["w", "prec", "ids", "step"]);
    assert_eq!(tensors[0].shape(), &[2, 3]);
    assert_eq!(tensors[0].dtype(), DType::F32);
    assert_eq!(tensors[0].to_vec(), f.to_vec());
    assert_eq!(tensors[1].dtype(), DType::F64);
    assert_eq!(tensors[1].to_vec_f64(), d.to_vec_f64());
    assert_eq!(tensors[2].dtype(), DType::I64);
    assert_eq!(tensors[2].to_vec_i64(), i.to_vec_i64());
    assert_eq!(tensors[3].shape(), &[] as &[usize]);
    assert_eq!(tensors[3].item(), 7.5);
}

#[test]
fn roundtrip_through_a_file() {
    let path = tmp_path("roundtrip");
    let t = Tensor::from_vec(vec![0.5, 1.5, 2.5, 3.5], &[4]).unwrap();
    save_safetensors(&path, &[("t", &t)]).unwrap();
    let got = load_safetensors(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(got[0].0, "t");
    assert_eq!(got[0].1.to_vec(), t.to_vec());
}

#[test]
fn golden_file_layout() {
    // Byte-level checks against the format spec: little-endian header length,
    // spaces padding the header to an 8-byte boundary, then raw data.
    let t = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let bytes = to_safetensors_bytes(&[("a", &t)]).unwrap();
    let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    assert_eq!((8 + hlen) % 8, 0);
    let header = std::str::from_utf8(&bytes[8..8 + hlen]).unwrap();
    assert_eq!(
        header.trim_end(),
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#
    );
    assert_eq!(
        &bytes[8 + hlen..],
        [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat()
    );
}

#[test]
fn loads_a_foreign_header_with_metadata_and_whitespace() {
    // Written by another producer: __metadata__, spaces, unordered fields.
    let header = r#"{ "__metadata__": {"format": "pt"}, "x": {"data_offsets": [0, 4], "shape": [1], "dtype": "F32"} }"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&2.5f32.to_le_bytes());
    let got = from_safetensors_bytes(&bytes).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].0, "x");
    assert_eq!(got[0].1.to_vec(), vec![2.5]);
}

#[test]
fn rejects_malformed_files() {
    let with_header = |h: &str, data: &[u8]| {
        let mut b = (h.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(h.as_bytes());
        b.extend_from_slice(data);
        b
    };
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("truncated length", vec![1, 2, 3]),
        ("header past eof", 999u64.to_le_bytes().to_vec()),
        ("not json", with_header("hello", &[])),
        ("not an object", with_header("[1]", &[])),
        (
            "missing offsets",
            with_header(
                r#"{"a":{"dtype":"F32","shape":[1]}}"#,
                &1.0f32.to_le_bytes(),
            ),
        ),
        (
            "offsets shape mismatch",
            with_header(
                r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
                &1.0f32.to_le_bytes(),
            ),
        ),
        (
            "offsets past data",
            with_header(
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                &[],
            ),
        ),
        ("trailing garbage after json", with_header(r#"{} x"#, &[])),
    ];
    for (name, bytes) in cases {
        assert!(
            matches!(from_safetensors_bytes(&bytes), Err(Error::Format { .. })),
            "case {name} did not error"
        );
    }

    let unsupported = with_header(
        r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
        &[0, 0, 0, 0],
    );
    assert!(matches!(
        from_safetensors_bytes(&unsupported),
        Err(Error::Unsupported { .. })
    ));
}

#[test]
fn rejects_duplicate_names_and_missing_paths() {
    let t = Tensor::scalar(1.0);
    assert!(matches!(
        to_safetensors_bytes(&[("a", &t), ("a", &t)]),
        Err(Error::Format { .. })
    ));
    assert!(matches!(
        load_safetensors("/nonexistent/ferro.safetensors"),
        Err(Error::Io { .. })
    ));
}

#[test]
fn escaped_names_roundtrip() {
    let t = Tensor::scalar(3.0);
    let name = "layer.0/w\"q\"\\attn\tend";
    let bytes = to_safetensors_bytes(&[(name, &t)]).unwrap();
    let got = from_safetensors_bytes(&bytes).unwrap();
    assert_eq!(got[0].0, name);
    assert_eq!(got[0].1.item(), 3.0);
}
