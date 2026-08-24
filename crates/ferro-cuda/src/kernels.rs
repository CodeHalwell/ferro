//! Host-side CUDA C source generation for the elementwise kernels. Pure
//! string work so it is unit-testable without a GPU; the sources are compiled
//! with nvrtc at first use on a real device (see `CudaBackend::get_kernel`).

use ferro_core::dispatch::ReduceKind;
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
        // Not fmaxf: it drops NaN, torch's relu propagates it.
        UnaryKind::Relu => "((v > 0.0f || isnan(v)) ? v : 0.0f)".to_string(),
        UnaryKind::Exp => "expf(v)".to_string(),
        UnaryKind::Sigmoid => "1.0f / (1.0f + expf(-v))".to_string(),
        UnaryKind::Tanh => "tanhf(v)".to_string(),
        UnaryKind::Sqrt => "sqrtf(v)".to_string(),
        UnaryKind::Abs => "fabsf(v)".to_string(),
        UnaryKind::Log => "logf(v)".to_string(),
        UnaryKind::Powf(p) => format!("powf(v, {})", c_f32(p)),
        // max-then-min chain matches core's CpuBackend (and torch):
        // min > max yields max everywhere; NaN passes through explicitly
        // since fmaxf/fminf would drop it.
        UnaryKind::Clamp { min, max } => {
            format!("(isnan(v) ? v : fminf(fmaxf(v, {}), {}))", c_f32(min), c_f32(max))
        }
        UnaryKind::Gtz => "(v > 0.0f) ? 1.0f : 0.0f".to_string(),
        // Tanh-approximation GELU, matching core's ops_ext/gelu.rs constants.
        UnaryKind::Gelu => {
            "0.5f * v * (1.0f + tanhf(0.7978846f * (v + 0.044715f * v * v * v)))".to_string()
        }
        UnaryKind::Silu => "v / (1.0f + expf(-v))".to_string(),
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

/// Broadcasting binary kernel, generated per (kind, rank): shapes and strides
/// are plain `unsigned int` kernel parameters (rank of them each), so one
/// compiled function serves every shape of that rank -- the source-text cache
/// in `get_kernel` therefore effectively keys on (kind, rank). Each thread
/// decomposes its flat output index into coordinates via divmod (innermost
/// dim first) and accumulates the two input offsets from the padded strides
/// (0 for broadcast dims, see [`broadcast_strides`]).
pub fn binary_bc_source(kind: BinaryKind, rank: usize) -> String {
    assert!(rank >= 1, "rank-0 outputs are padded to rank 1 by the caller");
    let params: String = (0..rank)
        .map(|d| format!(", unsigned int d{d}"))
        .chain((0..rank).map(|d| format!(", unsigned int sa{d}")))
        .chain((0..rank).map(|d| format!(", unsigned int sb{d}")))
        .collect();
    let divmod: String = (0..rank)
        .rev()
        .map(|d| format!("    c = rem % d{d}; rem /= d{d}; ia += c * sa{d}; ib += c * sb{d};\n"))
        .collect();
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(const float* a, const float* b, float* out, unsigned int n{params}) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int rem = i;
    unsigned int ia = 0;
    unsigned int ib = 0;
    unsigned int c;
{divmod}    float x = a[ia]; float y = b[ib];
    out[i] = {};
}}
"#,
        binary_expr(kind)
    )
}

/// Threads per block for the two-pass full reduction (and its finalize pass).
pub const REDUCE_BLOCK: u32 = 256;

/// Upper bound on first-pass blocks; beyond this each thread grid-strides
/// over multiple chunks so the partials buffer stays small and the single-
/// block finalize pass stays cheap.
pub const REDUCE_MAX_BLOCKS: u32 = 2048;

/// Pass 1 of the full reduction: one partial sum per block. Each thread
/// grid-strides over coalesced chunks, the block reduces via a shared-memory
/// binary tree, and thread 0 writes the block total to out[blockIdx.x].
pub fn reduce_partial_source() -> String {
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(const float* x, float* out, unsigned int n) {{
    __shared__ float sh[{REDUCE_BLOCK}];
    unsigned int tid = threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) acc += x[i];
    sh[tid] = acc;
    __syncthreads();
    for (unsigned int s = {REDUCE_BLOCK} / 2; s > 0; s >>= 1) {{
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }}
    if (tid == 0) out[blockIdx.x] = sh[0];
}}
"#
    )
}

/// Pass 2: one block reduces the per-block partials into out[0]. `n` is the
/// partial count, `total` the original element count (Mean divides by it).
pub fn reduce_finalize_source(kind: ReduceKind) -> String {
    let finish = match kind {
        ReduceKind::Sum => "out[0] = sh[0];",
        // Empty-input mean matches core's CPU path and torch: 0/0 = NaN.
        ReduceKind::Mean => "out[0] = sh[0] / (float)total;",
    };
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(const float* p, float* out, unsigned int n, unsigned int total) {{
    __shared__ float sh[{REDUCE_BLOCK}];
    unsigned int tid = threadIdx.x;
    float acc = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x) acc += p[i];
    sh[tid] = acc;
    __syncthreads();
    for (unsigned int s = {REDUCE_BLOCK} / 2; s > 0; s >>= 1) {{
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }}
    if (tid == 0) {{ {finish} }}
}}
"#
    )
}

/// Launch geometry for [`reduce_partial_source`]: ceil(n/block) blocks,
/// capped at [`REDUCE_MAX_BLOCKS`].
pub fn reduce_grid(n: usize) -> u32 {
    ((n as u32).div_ceil(REDUCE_BLOCK)).min(REDUCE_MAX_BLOCKS)
}

/// Sum over one dim via outer/inner decomposition (the FakeDevice reference
/// scheme): one thread per keepdim-layout output element, looping the reduced
/// extent. Output element (o, i) sums x[(o*red + k)*inner + i] over k.
pub fn sum_dim_source() -> String {
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(const float* x, float* out, unsigned int n, unsigned int red, unsigned int inner) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int o = i / inner;
    unsigned int r = i % inner;
    float acc = 0.0f;
    for (unsigned int k = 0; k < red; ++k) acc += x[(o * red + k) * inner + r];
    out[i] = acc;
}}
"#
    )
}

/// Row gather for embedding/index_select_t: one thread per output element.
/// out[i] = w[idx[i / inner] * inner + i % inner]; idx entries were
/// bounds-checked on the host before launch, so no guard is needed.
pub fn gather_source() -> String {
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(const long long* idx, const float* w, float* out, unsigned int inner, unsigned int n) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int o = i / inner;
    out[i] = w[idx[o] * inner + (i % inner)];
}}
"#
    )
}

/// Constant fill; the value is a kernel parameter so one compiled function
/// serves every fill.
pub fn fill_source() -> String {
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(float* out, unsigned int n, float value) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}}
"#
    )
}

/// Softmax pass 1, one block per row of `cols` elements: block-reduce the row
/// max via shared memory, then (reusing the buffer) block-reduce the sum of
/// exp(x - max). Writes stats[2*row] = max, stats[2*row+1] = sum so the apply
/// pass is a pure elementwise kernel -- two launches total, no host traffic.
pub fn softmax_row_stats_source() -> String {
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(const float* x, float* stats, unsigned int cols) {{
    __shared__ float sh[{REDUCE_BLOCK}];
    const float* xr = x + blockIdx.x * cols;
    unsigned int tid = threadIdx.x;
    float m = __int_as_float(0xff800000);
    for (unsigned int k = tid; k < cols; k += blockDim.x) m = fmaxf(m, xr[k]);
    sh[tid] = m;
    __syncthreads();
    for (unsigned int s = {REDUCE_BLOCK} / 2; s > 0; s >>= 1) {{
        if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]);
        __syncthreads();
    }}
    m = sh[0];
    __syncthreads();
    float acc = 0.0f;
    for (unsigned int k = tid; k < cols; k += blockDim.x) acc += expf(xr[k] - m);
    sh[tid] = acc;
    __syncthreads();
    for (unsigned int s = {REDUCE_BLOCK} / 2; s > 0; s >>= 1) {{
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }}
    if (tid == 0) {{
        stats[2 * blockIdx.x] = m;
        stats[2 * blockIdx.x + 1] = sh[0];
    }}
}}
"#
    )
}

/// Softmax pass 2: elementwise apply of the per-row stats. `log` selects
/// log_softmax (y = x - (m + log s)) over plain softmax (y = exp(x - m) / s).
pub fn softmax_apply_source(log: bool) -> String {
    let body = if log { "x[i] - (m + logf(s))" } else { "expf(x[i] - m) / s" };
    format!(
        r#"extern "C" __global__ void {KERNEL_NAME}(const float* x, const float* stats, float* out, unsigned int n, unsigned int cols) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int row = i / cols;
    float m = stats[2 * row];
    float s = stats[2 * row + 1];
    out[i] = {body};
}}
"#
    )
}

/// Host-side companion of [`sum_dim_source`]: element strides for indexing
/// a contiguous buffer of `in_shape` as if broadcast (numpy right-aligned
/// rules) to `out_shape`. Padded/size-1 dims get stride 0. Returns
/// `max(out_shape.len(), 1)` entries so rank-0 outputs still launch a rank-1
/// kernel over a single element.
pub fn broadcast_strides(in_shape: &[usize], out_shape: &[usize]) -> Vec<u32> {
    let rank = out_shape.len().max(1);
    let mut strides = vec![0u32; rank];
    let pad = rank - in_shape.len();
    let mut stride = 1u32;
    for d in (0..in_shape.len()).rev() {
        if in_shape[d] != 1 {
            strides[pad + d] = stride;
        }
        stride *= in_shape[d] as u32;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unary_exprs_cover_every_kind() {
        assert_eq!(unary_expr(UnaryKind::Neg), "-v");
        assert_eq!(unary_expr(UnaryKind::Relu), "((v > 0.0f || isnan(v)) ? v : 0.0f)");
        assert_eq!(unary_expr(UnaryKind::Exp), "expf(v)");
        assert_eq!(unary_expr(UnaryKind::Sigmoid), "1.0f / (1.0f + expf(-v))");
        assert_eq!(unary_expr(UnaryKind::Tanh), "tanhf(v)");
        assert_eq!(unary_expr(UnaryKind::Sqrt), "sqrtf(v)");
        assert_eq!(unary_expr(UnaryKind::Abs), "fabsf(v)");
        assert_eq!(unary_expr(UnaryKind::Log), "logf(v)");
        assert_eq!(unary_expr(UnaryKind::Powf(2.5)), "powf(v, 2.5f)");
        let clamp = UnaryKind::Clamp { min: -1.0, max: 2.0 };
        assert_eq!(unary_expr(clamp), "(isnan(v) ? v : fminf(fmaxf(v, -1.0f), 2.0f))");
        assert_eq!(unary_expr(UnaryKind::Gtz), "(v > 0.0f) ? 1.0f : 0.0f");
        assert_eq!(
            unary_expr(UnaryKind::Gelu),
            "0.5f * v * (1.0f + tanhf(0.7978846f * (v + 0.044715f * v * v * v)))"
        );
        assert_eq!(unary_expr(UnaryKind::Silu), "v / (1.0f + expf(-v))");
    }

    #[test]
    fn softmax_sources_form_a_two_pass_pipeline() {
        let stats = softmax_row_stats_source();
        assert!(stats.contains(r#"extern "C" __global__ void ferro_kernel(const float* x, float* stats, unsigned int cols)"#));
        assert!(stats.contains("stats[2 * blockIdx.x] = m;"));
        assert!(stats.contains("stats[2 * blockIdx.x + 1] = sh[0];"));
        // log_softmax and softmax share the stats pass but differ in apply.
        let (soft, lsoft) = (softmax_apply_source(false), softmax_apply_source(true));
        assert!(soft.contains("out[i] = expf(x[i] - m) / s;"));
        assert!(lsoft.contains("out[i] = x[i] - (m + logf(s));"));
        assert_ne!(soft, lsoft);
    }

    #[test]
    fn scalar_formatting_handles_nonfinite_and_exponents() {
        let one_sided = UnaryKind::Clamp { min: 0.0, max: f32::INFINITY };
        assert_eq!(unary_expr(one_sided), "(isnan(v) ? v : fminf(fmaxf(v, 0.0f), __int_as_float(0x7f800000)))");
        let lo = UnaryKind::Clamp { min: f32::NEG_INFINITY, max: 1e10 };
        assert_eq!(unary_expr(lo), "(isnan(v) ? v : fminf(fmaxf(v, __int_as_float(0xff800000)), 10000000000.0f))");
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
        assert!(src.contains("out[i] = ((v > 0.0f || isnan(v)) ? v : 0.0f);"));
        let src = binary_source(BinaryKind::Div);
        assert!(src.contains(r#"extern "C" __global__ void ferro_kernel(const float* a, const float* b, float* out, unsigned int n)"#));
        assert!(src.contains("out[i] = x / y;"));
    }

    #[test]
    fn gtz_source_is_the_relu_gradient_mask() {
        let src = unary_source(UnaryKind::Gtz);
        assert!(src.contains("out[i] = (v > 0.0f) ? 1.0f : 0.0f;"));
    }

    #[test]
    fn broadcast_binary_source_decomposes_flat_index() {
        let src = binary_bc_source(BinaryKind::Add, 2);
        assert!(src.contains(r#"extern "C" __global__ void ferro_kernel(const float* a, const float* b, float* out, unsigned int n, unsigned int d0, unsigned int d1, unsigned int sa0, unsigned int sa1, unsigned int sb0, unsigned int sb1)"#));
        // Innermost dim is peeled first.
        let d1 = src.find("c = rem % d1").unwrap();
        let d0 = src.find("c = rem % d0").unwrap();
        assert!(d1 < d0);
        assert!(src.contains("ia += c * sa1; ib += c * sb1;"));
        assert!(src.contains("out[i] = x + y;"));
        // Rank (not shape) is baked into the source, so the cache keys on it.
        assert_ne!(binary_bc_source(BinaryKind::Add, 1), src);
        assert_ne!(binary_bc_source(BinaryKind::Mul, 2), src);
    }

    #[test]
    fn reduce_sources_form_a_two_pass_pipeline() {
        let partial = reduce_partial_source();
        assert!(partial.contains(r#"extern "C" __global__ void ferro_kernel(const float* x, float* out, unsigned int n)"#));
        assert!(partial.contains("__shared__ float sh[256];"));
        assert!(partial.contains("out[blockIdx.x] = sh[0];"));
        // Finalize is generated per kind and divides by the original count for Mean.
        assert_ne!(reduce_finalize_source(ReduceKind::Sum), reduce_finalize_source(ReduceKind::Mean));
        assert!(reduce_finalize_source(ReduceKind::Sum).contains("out[0] = sh[0];"));
        assert!(reduce_finalize_source(ReduceKind::Mean).contains("out[0] = sh[0] / (float)total;"));
    }

    #[test]
    fn reduce_grid_caps_blocks_and_handles_small_n() {
        assert_eq!(reduce_grid(0), 0);
        assert_eq!(reduce_grid(1), 1);
        assert_eq!(reduce_grid(255), 1);
        assert_eq!(reduce_grid(256), 1);
        assert_eq!(reduce_grid(257), 2);
        assert_eq!(reduce_grid(u32::MAX as usize), 2048);
    }

    #[test]
    fn sum_dim_source_uses_outer_inner_decomposition() {
        let src = sum_dim_source();
        assert!(src.contains(r#"extern "C" __global__ void ferro_kernel(const float* x, float* out, unsigned int n, unsigned int red, unsigned int inner)"#));
        assert!(src.contains("acc += x[(o * red + k) * inner + r];"));
    }

    #[test]
    fn fill_source_writes_the_parameter_value() {
        let src = fill_source();
        assert!(src.contains(r#"extern "C" __global__ void ferro_kernel(float* out, unsigned int n, float value)"#));
        assert!(src.contains("out[i] = value;"));
    }

    #[test]
    fn broadcast_strides_follow_numpy_rules() {
        assert_eq!(broadcast_strides(&[2, 3], &[2, 3]), vec![3, 1]);
        assert_eq!(broadcast_strides(&[3], &[2, 3]), vec![0, 1]);
        assert_eq!(broadcast_strides(&[2, 1], &[2, 3]), vec![1, 0]);
        assert_eq!(broadcast_strides(&[], &[2, 3]), vec![0, 0]);
        assert_eq!(broadcast_strides(&[1, 3], &[4, 2, 3]), vec![0, 0, 1]);
        // Rank-0 output is padded to a single rank-1 element.
        assert_eq!(broadcast_strides(&[], &[]), vec![0]);
    }

    // Host simulation of the generated kernel's index arithmetic: divmod over
    // the output shape with padded strides must reproduce numpy broadcasting.
    #[test]
    fn broadcast_index_math_matches_reference() {
        let (sa, sb, out) = (vec![2usize, 1], vec![3usize], vec![2usize, 3]);
        let a = [10.0f32, 20.0];
        let b = [1.0f32, 2.0, 3.0];
        let stra = broadcast_strides(&sa, &out);
        let strb = broadcast_strides(&sb, &out);
        let n: usize = out.iter().product();
        let mut got = Vec::new();
        for i in 0..n {
            let (mut rem, mut ia, mut ib) = (i as u32, 0u32, 0u32);
            for d in (0..out.len()).rev() {
                let c = rem % out[d] as u32;
                rem /= out[d] as u32;
                ia += c * stra[d];
                ib += c * strb[d];
            }
            got.push(a[ia as usize] + b[ib as usize]);
        }
        assert_eq!(got, vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn parametrized_kinds_generate_distinct_sources() {
        assert_ne!(unary_source(UnaryKind::Powf(2.0)), unary_source(UnaryKind::Powf(3.0)));
        let a = UnaryKind::Clamp { min: 0.0, max: 1.0 };
        let b = UnaryKind::Clamp { min: 0.0, max: 2.0 };
        assert_ne!(unary_source(a), unary_source(b));
    }
}
