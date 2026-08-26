//! Bit-level IEEE binary16 (f16) and bfloat16 conversions, dependency-free.
//! Storage keeps raw `u16` bit patterns (`Storage::F16`/`Storage::BF16`);
//! these functions are the only translation in and out of them. All
//! f32 -> half conversions round to nearest, ties to even, saturating to the
//! signed infinity on overflow and quieting NaNs while preserving the top
//! payload bits - the same behavior torch and the safetensors ecosystem
//! exhibit. (amp::bf16_round is the separate f32-domain rounding used by
//! autocast; it deliberately passes NaNs through with their full payload.)

/// bf16 -> f32: exact (bf16 is f32's top half).
pub fn f32_from_bf16_bits(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 -> bf16 bits, round to nearest even; overflow saturates to inf.
pub fn bf16_bits_from_f32(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        // Quiet NaN keeping the sign and top payload bits; the forced bit
        // guarantees a payload whose top half was zero stays a NaN.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let sign = (bits >> 16) as u16 & 0x8000;
    let mag = bits & 0x7FFF_FFFF;
    // RNE via integer add: 0x7FFF plus the kept LSB carries exactly when the
    // dropped half exceeds a tie, or ties into an odd kept LSB. The carry
    // ripples into the exponent, which is precisely round-up to the next
    // binade - and past the largest finite bf16, into infinity.
    let rounded = (mag + 0x7FFF + ((mag >> 16) & 1)) >> 16;
    if rounded >= 0x7F80 {
        return sign | 0x7F80;
    }
    sign | rounded as u16
}

/// f16 -> f32: exact for every finite value, infinity, and NaN (payload
/// shifts up; quietness is preserved since the quiet bit shifts with it).
pub fn f32_from_f16_bits(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x03FF) as u32;
    match exp {
        // Zero or subnormal: value is mant * 2^-24, exact in f32.
        0 => {
            let mag = mant as f32 * f32::from_bits(0x3380_0000); // 2^-24
            if sign != 0 {
                -mag
            } else {
                mag
            }
        }
        0x1F => f32::from_bits(sign | 0x7F80_0000 | (mant << 13)),
        _ => f32::from_bits(sign | ((exp as u32 + 112) << 23) | (mant << 13)),
    }
}

/// f32 -> f16 bits, round to nearest even, with gradual underflow to f16
/// subnormals; overflow (anything rounding past 65504) saturates to inf.
pub fn f16_bits_from_f32(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mag = bits & 0x7FFF_FFFF;
    if mag > 0x7F80_0000 {
        // NaN: quiet, keep the top 10 payload bits.
        return sign | 0x7C00 | 0x0200 | ((mag >> 13) & 0x03FF) as u16;
    }
    if mag < 0x3880_0000 {
        // |v| below the smallest normal f16 (2^-14): the result is the
        // subnormal count of 2^-24 steps. |v| * 2^24 is exact (a pure
        // exponent shift), so one round_ties_even is the single correct
        // rounding; 1024 overflows into the smallest normal encoding, which
        // is exactly what bit pattern 0x0400 means.
        let steps = f32::from_bits(mag) * 16_777_216.0; // 2^24
        return sign | steps.round_ties_even() as u16;
    }
    // Normal range: RNE on the 13 dropped mantissa bits via the same
    // carry-into-exponent add as bf16; a carry past exponent 30 is overflow.
    let rounded = mag + 0x0FFF + ((mag >> 13) & 1);
    let exp16 = ((rounded >> 23) as i32) - 112;
    if exp16 >= 0x1F {
        return sign | 0x7C00;
    }
    sign | ((exp16 as u16) << 10) | ((rounded >> 13) & 0x03FF) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_bit_patterns() {
        // (bits, value) pairs from the IEEE binary16 tables.
        let cases: &[(u16, f32)] = &[
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x3C00, 1.0),
            (0xBC00, -1.0),
            (0x4100, 2.5),
            (0x7BFF, 65504.0),   // largest finite
            (0x0400, 6.1035156e-5), // smallest normal, 2^-14
            (0x0001, 5.9604645e-8), // smallest subnormal, 2^-24
            (0x3555, 0.33325195), // nearest f16 to 1/3
        ];
        for &(bits, val) in cases {
            assert_eq!(f32_from_f16_bits(bits), val, "decode {bits:#06x}");
            assert_eq!(f16_bits_from_f32(val), bits, "encode {val}");
        }
        assert_eq!(f32_from_f16_bits(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f32_from_f16_bits(0x7C00), f32::INFINITY);
        assert_eq!(f32_from_f16_bits(0xFC00), f32::NEG_INFINITY);
        assert!(f32_from_f16_bits(0x7E00).is_nan());
    }

    #[test]
    fn f16_rounding_is_nearest_even_and_saturating() {
        // 1 + 2^-11 sits exactly between 1.0 (0x3C00) and the next f16
        // (0x3C01): the tie must go to the even mantissa.
        assert_eq!(f16_bits_from_f32(1.0 + 0.00048828125), 0x3C00);
        // 1 + 3*2^-12 is between the same neighbors but past the midpoint.
        assert_eq!(f16_bits_from_f32(1.0 + 0.000732421875), 0x3C01);
        // The overflow cutoff: 65519.996 rounds down to 65504, 65520 to inf.
        assert_eq!(f16_bits_from_f32(65519.0), 0x7BFF);
        assert_eq!(f16_bits_from_f32(65520.0), 0x7C00);
        assert_eq!(f16_bits_from_f32(f32::INFINITY), 0x7C00);
        assert_eq!(f16_bits_from_f32(f32::NEG_INFINITY), 0xFC00);
        assert!(f32_from_f16_bits(f16_bits_from_f32(f32::NAN)).is_nan());
        // Subnormal rounding: half the smallest subnormal ties to zero
        // (even), three quarters rounds up to one step.
        assert_eq!(f16_bits_from_f32(2.9802322e-8), 0x0000);
        assert_eq!(f16_bits_from_f32(4.4703484e-8), 0x0001);
        // Just below the normal threshold rounds up into the first normal.
        assert_eq!(f16_bits_from_f32(6.10333e-5), 0x0400);
    }

    #[test]
    fn f16_round_trips_every_finite_value_exactly() {
        for bits in 0u16..=0xFFFF {
            let exp = (bits >> 10) & 0x1F;
            if exp == 0x1F {
                continue; // inf/NaN checked separately
            }
            assert_eq!(
                f16_bits_from_f32(f32_from_f16_bits(bits)),
                bits,
                "{bits:#06x}"
            );
        }
    }

    #[test]
    fn bf16_is_the_top_half_of_f32() {
        for v in [0.0f32, -0.0, 1.0, -2.5, 3.3895314e38, 1e-40] {
            let bits = bf16_bits_from_f32(v);
            assert_eq!(f32_from_bf16_bits(bits).to_bits() >> 16, bits as u32);
        }
        // RNE at the bf16 tie: 1 + 2^-8 is exactly between 1.0 (0x3F80) and
        // the next bf16 (0x3F81); ties to even keeps 0x3F80. Past the
        // midpoint (1 + 2^-8 + 2^-9) rounds up, and the tie above an odd
        // mantissa (1 + 3*2^-8) rounds up to even 0x3F82.
        assert_eq!(bf16_bits_from_f32(1.00390625), 0x3F80);
        assert_eq!(bf16_bits_from_f32(1.005859375), 0x3F81);
        assert_eq!(bf16_bits_from_f32(1.01171875), 0x3F82);
        // Overflow past bf16 max (~3.39e38) saturates.
        assert_eq!(bf16_bits_from_f32(f32::MAX), 0x7F80);
        assert_eq!(bf16_bits_from_f32(-f32::MAX), 0xFF80);
        assert!(f32_from_bf16_bits(bf16_bits_from_f32(f32::NAN)).is_nan());
    }

    #[test]
    fn bf16_round_trips_every_finite_value_exactly() {
        for hi in 0u16..=0xFFFF {
            let exp = (hi >> 7) & 0xFF;
            if exp == 0xFF {
                continue;
            }
            assert_eq!(bf16_bits_from_f32(f32_from_bf16_bits(hi)), hi, "{hi:#06x}");
        }
    }
}
