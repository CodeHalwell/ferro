//! Host-side CUDA C source generation for the elementwise kernels. Pure
//! string work so it is unit-testable without a GPU; the sources are compiled
//! with nvrtc at first use on a real device (see `CudaBackend::get_kernel`).

use ferro_core::{BinaryKind, UnaryKind};

/// Every generated module exports exactly one kernel under this name; the
/// function cache is keyed by source text, so the name never collides.
pub const KERNEL_NAME: &str = "ferro_kernel";

/// Format an f32 as a CUDA C expression. Non-finite values (legal in e.g.
/// one-sided Clamp) have no literal spelling, so spell them as bit patterns.
fn c_f32(v: f32) -> String {
    if v.is_nan() {
        "__int_as_float(0x7fc00000)".to_string()
    } else if v == f32::INFINITY {
        "__int_as_float(0x7f800000)".to_string()
    } else if v == f32::NEG_INFINITY {
        "__int_as_float(0xff800000)".to_string()
    } else {
        format!("{v:?}f")
    }
}

/// CUDA C expression computing `kind` of the input element `v`.
pub fn unary_expr(kind: UnaryKind) -> String {
    match kind {
        UnaryKind::Neg => "-v".to_string(),
        UnaryKind::Relu => "fmaxf(v, 0.0f)".to_string(),
        UnaryKind::Exp => "expf(v)".to_string(),
        UnaryKind::Sigmoid => "1.0f / (1.0f + expf(-v))".to_string(),
        UnaryKind::Tanh => "tanhf(v)".to_string(),
        UnaryKind::Sqrt => "sqrtf(v)".to_string(),
        UnaryKind::Abs => "fabsf(v)".to_string(),
        UnaryKind::Log => "logf(v)".to_string(),
        UnaryKind::Powf(p) => format!("powf(v, {})", c_f32(p)),
        // max-then-min chain matches core's CpuBackend (and torch):
        // min > max yields max everywhere.
        UnaryKind::Clamp { min, max } => {
            format!("fminf(fmaxf(v, {}), {})", c_f32(min), c_f32(max))
        }
    }
}

/// CUDA C expression combining elements `x` (from a) and `y` (from b).
pub fn binary_expr(kind: BinaryKind) -> &'static str {
    match kind {
        BinaryKind::Add => "x + y",
        BinaryKind::Sub => "x - y",
        BinaryKind::Mul => "x * y",
        BinaryKind::Div => "x / y",
    }
}

pub fn unary_source(kind: UnaryKind) -> String {
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(const float* x, float* out, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{ float v = x[i]; out[i] = {}; }}
}}
"#,
        unary_expr(kind)
    )
}

pub fn binary_source(kind: BinaryKind) -> String {
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(const float* a, const float* b, float* out, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{ float x = a[i]; float y = b[i]; out[i] = {}; }}
}}
"#,
        binary_expr(kind)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_exprs_cover_every_kind() {
        assert_eq!(unary_expr(UnaryKind::Neg), "-v");
        assert_eq!(unary_expr(UnaryKind::Relu), "fmaxf(v, 0.0f)");
        assert_eq!(unary_expr(UnaryKind::Exp), "expf(v)");
        assert_eq!(unary_expr(UnaryKind::Sigmoid), "1.0f / (1.0f + expf(-v))");
        assert_eq!(unary_expr(UnaryKind::Tanh), "tanhf(v)");
        assert_eq!(unary_expr(UnaryKind::Sqrt), "sqrtf(v)");
        assert_eq!(unary_expr(UnaryKind::Abs), "fabsf(v)");
        assert_eq!(unary_expr(UnaryKind::Log), "logf(v)");
        assert_eq!(unary_expr(UnaryKind::Powf(2.5)), "powf(v, 2.5f)");
        let clamp = UnaryKind::Clamp { min: -1.0, max: 2.0 };
        assert_eq!(unary_expr(clamp), "fminf(fmaxf(v, -1.0f), 2.0f)");
    }

    #[test]
    fn scalar_formatting_handles_nonfinite_and_exponents() {
        let one_sided = UnaryKind::Clamp { min: 0.0, max: f32::INFINITY };
        assert_eq!(unary_expr(one_sided), "fminf(fmaxf(v, 0.0f), __int_as_float(0x7f800000))");
        let lo = UnaryKind::Clamp { min: f32::NEG_INFINITY, max: 1e10 };
        assert_eq!(unary_expr(lo), "fminf(fmaxf(v, __int_as_float(0xff800000)), 10000000000.0f)");
        assert_eq!(unary_expr(UnaryKind::Powf(f32::NAN)), "powf(v, __int_as_float(0x7fc00000))");
    }

    #[test]
    fn binary_exprs_cover_every_kind() {
        assert_eq!(binary_expr(BinaryKind::Add), "x + y");
        assert_eq!(binary_expr(BinaryKind::Sub), "x - y");
        assert_eq!(binary_expr(BinaryKind::Mul), "x * y");
        assert_eq!(binary_expr(BinaryKind::Div), "x / y");
    }

    #[test]
    fn sources_declare_the_exported_kernel() {
        let src = unary_source(UnaryKind::Relu);
        assert!(src.contains(r#"extern "C" __global__ void ferro_kernel(const float* x, float* out, unsigned int n)"#));
        assert!(src.contains("out[i] = fmaxf(v, 0.0f);"));
        let src = binary_source(BinaryKind::Div);
        assert!(src.contains(r#"extern "C" __global__ void ferro_kernel(const float* a, const float* b, float* out, unsigned int n)"#));
        assert!(src.contains("out[i] = x / y;"));
    }

    #[test]
    fn parametrized_kinds_generate_distinct_sources() {
        assert_ne!(unary_source(UnaryKind::Powf(2.0)), unary_source(UnaryKind::Powf(3.0)));
        let a = UnaryKind::Clamp { min: 0.0, max: 1.0 };
        let b = UnaryKind::Clamp { min: 0.0, max: 2.0 };
        assert_ne!(unary_source(a), unary_source(b));
    }
}
