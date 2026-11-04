use ferro_core::dispatch::{Backend, UnaryKind};
use ferro_fastcpu::elementwise::FastCpuBackend;
#[allow(dead_code)]
mod golden { include!("../../ferro-core/tests/fixtures/clamp_bounds_goldens.rs");
    pub fn input() -> &'static [u32] { INPUT32 }
    pub fn cases() -> &'static [(u32, u32, &'static [u32])] { CLAMP32 }
}

#[test]
fn serial_chunk_and_parallel_clamp_match_torch_bits_for_nan_bounds_and_zero_ties() {
    let backend = FastCpuBackend;
    for &(min, max, expected) in golden::cases() { for count in [9, 17, 1031, 262_147] {
        let input = (0..count).map(|i| f32::from_bits(golden::input()[i % 9])).collect::<Vec<_>>();
        let output = backend.unary(UnaryKind::Clamp { min: f32::from_bits(min), max: f32::from_bits(max) }, &input);
        assert_eq!(output.len(), count);
        for (i, &value) in output.iter().enumerate() {
            let expected = expected[i % 9];
            assert!(value.to_bits() == expected || value.is_nan() && f32::from_bits(expected).is_nan(), "index {i} count {count} min {min:x} max {max:x}");
        }
    } }
}
