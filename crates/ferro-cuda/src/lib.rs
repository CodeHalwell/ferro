//! CUDA backend for ferro, built on `cudarc` with runtime library loading.
//!
//! Storage is device-resident (dispatcher phase 3): the `*_dev` methods take
//! and return [`CudaBuf`]s wrapping `CudaSlice<f32>`, so chained ops keep
//! their data in GPU memory. The full extended surface is implemented --
//! broadcasting binaries, full reductions, per-dim sums, fills, and
//! transpose-flagged matmul -- which is what core's device autograd needs to
//! run backward passes resident. The host-slice `Backend` methods remain as
//! the fallback path core uses for non-resident tensors; they are thin
//! wrappers (htod, `*_dev` compute, dtoh) over the same kernels.
//!
//! cudarc is configured with `dynamic-loading`: libcuda/libnvrtc/libcublas
//! are dlopened at runtime, so this crate compiles and its host-side tests
//! run on machines without any CUDA installation. [`install`] returns `Err`
//! (never panics) when no driver or device is present.

mod kernels;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ferro_core::dispatch::{DeviceBuffer, ReduceKind};
use ferro_core::{register_backend, Backend, BinaryKind, Device, Error, Result, UnaryKind};

/// Cheap detection: is the CUDA driver library (libcuda) loadable? True does
/// not guarantee a usable device (the driver can be present with zero GPUs);
/// [`install`] performs the real device init and reports failures as `Err`.
pub fn is_available() -> bool {
    unsafe { cudarc::driver::sys::is_culib_present() }
}

/// Map a cudarc failure into core's error type. The `*_dev` seam has a real
/// error channel, so driver errors surface as `Err` rather than panics.
fn cuda_err(op: &'static str, e: impl std::fmt::Display) -> Error {
    Error::Unsupported { op, msg: format!("CUDA error: {e}") }
}

/// Device-resident buffer: a `CudaSlice<f32>` tagged with the `Device` it
/// lives on. Core hands these back through `&dyn DeviceBuffer`; the backend
/// recovers the slice by downcasting via `as_any`.
pub struct CudaBuf {
    data: CudaSlice<f32>,
    device: Device,
}

impl DeviceBuffer for CudaBuf {
    fn device(&self) -> Device {
        self.device
    }
    fn len(&self) -> usize {
        self.data.len()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// `Backend` implementation with device-resident buffers ([`CudaBuf`]).
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    device: Device,
    // nvrtc-compiled elementwise kernels, keyed by generated source text so
    // parametrized kinds (Powf, Clamp) cache per scalar value.
    funcs: Mutex<HashMap<String, CudaFunction>>,
}

impl CudaBackend {
    /// Initialize device `ordinal`. Errors (rather than panicking) when the
    /// CUDA libraries are missing or the device cannot be initialized.
    pub fn new(ordinal: u32) -> std::result::Result<CudaBackend, String> {
        // cudarc's dynamic loader panics (not Err) on a missing library, so
        // probe all three libraries up front to keep this a clean Err path.
        unsafe {
            if !cudarc::driver::sys::is_culib_present() {
                return Err("CUDA driver library (libcuda) not found; no GPU driver installed".to_string());
            }
            if !cudarc::nvrtc::sys::is_culib_present() {
                return Err("NVRTC library (libnvrtc) not found; install the CUDA toolkit runtime".to_string());
            }
            if !cudarc::cublas::sys::is_culib_present() {
                return Err("cuBLAS library (libcublas) not found; install the CUDA toolkit runtime".to_string());
            }
        }
        let ctx = CudaContext::new(ordinal as usize)
            .map_err(|e| format!("failed to initialize CUDA device {ordinal}: {e}"))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|e| format!("failed to create cuBLAS handle: {e}"))?;
        let device = Device::Cuda(ordinal);
        Ok(CudaBackend { ctx, stream, blas, device, funcs: Mutex::new(HashMap::new()) })
    }

    /// Downcast a core-provided buffer back to this backend's `CudaBuf`,
    /// rejecting buffers from other backends or other CUDA devices.
    fn resident<'a>(&self, op: &'static str, buf: &'a dyn DeviceBuffer) -> Result<&'a CudaBuf> {
        let buf = buf.as_any().downcast_ref::<CudaBuf>().ok_or_else(|| Error::Unsupported {
            op,
            msg: "device buffer was not allocated by the CUDA backend".into(),
        })?;
        if buf.device != self.device {
            return Err(Error::Unsupported {
                op,
                msg: format!("buffer lives on {} but this backend serves {}", buf.device, self.device),
            });
        }
        Ok(buf)
    }

    fn wrap(&self, data: CudaSlice<f32>) -> Box<dyn DeviceBuffer> {
        Box::new(CudaBuf { data, device: self.device })
    }

    /// Fetch the cached kernel for `src`, compiling it with nvrtc on first
    /// use. nvrtc failures on our generated source are bugs, but they are
    /// still reported as `Err` so callers never bring the process down.
    fn get_kernel(&self, op: &'static str, src: &str) -> Result<CudaFunction> {
        let mut cache = self.funcs.lock().unwrap();
        if let Some(f) = cache.get(src) {
            return Ok(f.clone());
        }
        let ptx = compile_ptx(src)
            .map_err(|e| Error::Unsupported { op, msg: format!("nvrtc failed to compile kernel: {e}\nsource:\n{src}") })?;
        let module = self.ctx.load_module(ptx).map_err(|e| cuda_err(op, e))?;
        let f = module.load_function(kernels::KERNEL_NAME).map_err(|e| cuda_err(op, e))?;
        cache.insert(src.to_string(), f.clone());
        Ok(f)
    }

    /// Launch an elementwise kernel over device-resident inputs, writing to a
    /// freshly allocated output slice. No host round trip.
    fn launch_elementwise(&self, op: &'static str, src: &str, inputs: &[&CudaSlice<f32>], n: usize) -> Result<CudaSlice<f32>> {
        let func = self.get_kernel(op, src)?;
        let mut out = self.stream.alloc_zeros::<f32>(n).map_err(|e| cuda_err(op, e))?;
        // Zero-sized launches are invalid; an empty tensor's result is the
        // freshly allocated empty buffer.
        if n == 0 {
            return Ok(out);
        }
        let n_arg = n as u32;
        let mut launch = self.stream.launch_builder(&func);
        for d in inputs {
            launch.arg(*d);
        }
        launch.arg(&mut out);
        launch.arg(&n_arg);
        unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }.map_err(|e| cuda_err(op, e))?;
        Ok(out)
    }

    /// Logical row-major (m,k) @ (k,n) -> (m,n) over device-resident
    /// operands; `ta`/`tb` mark an operand as stored transposed.
    #[allow(clippy::too_many_arguments)] // mirrors the Backend::matmul_dev seam
    fn sgemm(&self, op: &'static str, a: &CudaSlice<f32>, b: &CudaSlice<f32>, m: usize, k: usize, n: usize, ta: bool, tb: bool) -> Result<CudaSlice<f32>> {
        let mut c = self.stream.alloc_zeros::<f32>(m * n).map_err(|e| cuda_err(op, e))?;
        if k > 0 {
            let cfg = row_major_sgemm_cfg(m, k, n, ta, tb);
            // Swapped operands: b is cuBLAS "A", a is cuBLAS "B".
            unsafe { self.blas.gemm(cfg, b, a, &mut c) }.map_err(|e| cuda_err(op, e))?;
        }
        Ok(c)
    }

    fn htod(&self, op: &'static str, data: &[f32]) -> Result<CudaSlice<f32>> {
        self.stream.clone_htod(data).map_err(|e| cuda_err(op, e))
    }

    fn dtoh(&self, op: &'static str, data: &CudaSlice<f32>) -> Result<Vec<f32>> {
        self.stream.clone_dtoh(data).map_err(|e| cuda_err(op, e))
    }
}

/// cuBLAS gemm parameters computing row-major C(m,n) = op(A)(m,k) * op(B)(k,n),
/// where `ta`/`tb` mean the operand buffer stores the transpose ((k,m)/(n,k)
/// row-major).
///
/// cuBLAS is column-major, so a row-major buffer of shape (r, c) is, viewed
/// column-major, the transposed (c, r) matrix. Rather than transposing, use
/// the identity C^T = op(B)^T * op(A)^T: an sgemm over the column-major views
/// with the operands swapped (B's buffer as cuBLAS "A", A's as cuBLAS "B")
/// and m/n swapped produces C^T column-major, whose bytes are exactly C
/// row-major. Per flag:
/// - !tb: B's buffer is (k,n) row-major = B^T column-major, which is already
///   op(B)^T -> CUBLAS_OP_N, lda = n (rows of the column-major view).
/// - tb: B's buffer is (n,k) row-major = B column-major; cuBLAS must
///   transpose it to get B^T -> CUBLAS_OP_T, lda = k.
/// - !ta: A's buffer is (m,k) row-major = A^T column-major -> CUBLAS_OP_N,
///   ldb = k.
/// - ta: A's buffer is (k,m) row-major = A column-major -> CUBLAS_OP_T,
///   ldb = m.
///
/// C^T is (n,m) column-major, so ldc = n.
fn row_major_sgemm_cfg(m: usize, k: usize, n: usize, ta: bool, tb: bool) -> GemmConfig<f32> {
    let flag = |t: bool| if t { cublasOperation_t::CUBLAS_OP_T } else { cublasOperation_t::CUBLAS_OP_N };
    GemmConfig {
        transa: flag(tb),
        transb: flag(ta),
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0,
        lda: if tb { k } else { n } as i32,
        ldb: if ta { m } else { k } as i32,
        beta: 0.0,
        ldc: n as i32,
    }
}

impl Backend for CudaBackend {
    // Host-slice fallbacks: htod, run the same device kernels as the *_dev
    // path, dtoh. The seam returns bare Vec<f32> (no error channel), so a
    // driver failure here can only panic; the *_dev methods return Err.
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }
        let xd = self.htod("unary", x).expect("htod copy failed");
        let out = self.launch_elementwise("unary", &kernels::unary_source(kind), &[&xd], x.len());
        self.dtoh("unary", &out.expect("kernel launch failed")).expect("dtoh copy failed")
    }

    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.is_empty() {
            return Vec::new();
        }
        let ad = self.htod("binary", a).expect("htod copy failed");
        let bd = self.htod("binary", b).expect("htod copy failed");
        let out = self.launch_elementwise("binary", &kernels::binary_source(kind), &[&ad, &bd], a.len());
        self.dtoh("binary", &out.expect("kernel launch failed")).expect("dtoh copy failed")
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        if m == 0 || n == 0 {
            return Vec::new();
        }
        let ad = self.htod("matmul", a).expect("htod copy failed");
        let bd = self.htod("matmul", b).expect("htod copy failed");
        let c = self.sgemm("matmul", &ad, &bd, m, k, n, false, false).expect("cublas sgemm failed");
        self.dtoh("matmul", &c).expect("dtoh copy failed")
    }

    fn alloc_from_host(&self, data: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        Ok(self.wrap(self.htod("alloc_from_host", data)?))
    }

    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        self.dtoh("copy_to_host", &self.resident("copy_to_host", buf)?.data)
    }

    fn unary_dev(&self, kind: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        let x = self.resident("unary_dev", x)?;
        let out = self.launch_elementwise("unary_dev", &kernels::unary_source(kind), &[&x.data], x.data.len())?;
        Ok(self.wrap(out))
    }

    fn binary_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        let a = self.resident("binary_dev", a)?;
        let b = self.resident("binary_dev", b)?;
        if a.data.len() != b.data.len() {
            return Err(Error::Unsupported {
                op: "binary_dev",
                msg: format!("length mismatch: {} vs {}", a.data.len(), b.data.len()),
            });
        }
        let src = kernels::binary_source(kind);
        let out = self.launch_elementwise("binary_dev", &src, &[&a.data, &b.data], a.data.len())?;
        Ok(self.wrap(out))
    }

    fn matmul_dev(&self, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer, m: usize, k: usize, n: usize, ta: bool, tb: bool) -> Result<Box<dyn DeviceBuffer>> {
        let a = self.resident("matmul_dev", a)?;
        let b = self.resident("matmul_dev", b)?;
        Ok(self.wrap(self.sgemm("matmul_dev", &a.data, &b.data, m, k, n, ta, tb)?))
    }

    fn binary_bc_dev(&self, kind: BinaryKind, a: &dyn DeviceBuffer, sa: &[usize], b: &dyn DeviceBuffer, sb: &[usize], out_shape: &[usize]) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "binary_bc_dev";
        let a = self.resident(OP, a)?;
        let b = self.resident(OP, b)?;
        // Rank-0 outputs launch as a single rank-1 element (see kernels.rs).
        let mut dims: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
        if dims.is_empty() {
            dims.push(1);
        }
        let stra = kernels::broadcast_strides(sa, out_shape);
        let strb = kernels::broadcast_strides(sb, out_shape);
        let n: usize = out_shape.iter().product();
        let func = self.get_kernel(OP, &kernels::binary_bc_source(kind, dims.len()))?;
        let mut out = self.stream.alloc_zeros::<f32>(n).map_err(|e| cuda_err(OP, e))?;
        if n > 0 {
            let n_arg = n as u32;
            let mut launch = self.stream.launch_builder(&func);
            launch.arg(&a.data);
            launch.arg(&b.data);
            launch.arg(&mut out);
            launch.arg(&n_arg);
            for d in dims.iter().chain(stra.iter()).chain(strb.iter()) {
                launch.arg(d);
            }
            unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }.map_err(|e| cuda_err(OP, e))?;
        }
        Ok(self.wrap(out))
    }

    fn reduce_dev(&self, kind: ReduceKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "reduce_dev";
        let x = self.resident(OP, x)?;
        let func = self.get_kernel(OP, &kernels::reduce_source(kind))?;
        let mut out = self.stream.alloc_zeros::<f32>(1).map_err(|e| cuda_err(OP, e))?;
        let n_arg = x.data.len() as u32;
        let mut launch = self.stream.launch_builder(&func);
        launch.arg(&x.data);
        launch.arg(&mut out);
        launch.arg(&n_arg);
        // The reduce kernel is single-threaded (see kernels.rs).
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        unsafe { launch.launch(cfg) }.map_err(|e| cuda_err(OP, e))?;
        Ok(self.wrap(out))
    }

    fn sum_dim_dev(&self, x: &dyn DeviceBuffer, shape: &[usize], dim: usize) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "sum_dim_dev";
        let x = self.resident(OP, x)?;
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();
        let n = outer * inner;
        let func = self.get_kernel(OP, &kernels::sum_dim_source())?;
        let mut out = self.stream.alloc_zeros::<f32>(n).map_err(|e| cuda_err(OP, e))?;
        if n > 0 {
            let (n_arg, red, inner) = (n as u32, shape[dim] as u32, inner as u32);
            let mut launch = self.stream.launch_builder(&func);
            launch.arg(&x.data);
            launch.arg(&mut out);
            launch.arg(&n_arg);
            launch.arg(&red);
            launch.arg(&inner);
            unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }.map_err(|e| cuda_err(OP, e))?;
        }
        Ok(self.wrap(out))
    }

    fn fill_dev(&self, value: f32, len: usize) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "fill_dev";
        let func = self.get_kernel(OP, &kernels::fill_source())?;
        let mut out = self.stream.alloc_zeros::<f32>(len).map_err(|e| cuda_err(OP, e))?;
        if len > 0 {
            let n_arg = len as u32;
            let mut launch = self.stream.launch_builder(&func);
            launch.arg(&mut out);
            launch.arg(&n_arg);
            launch.arg(&value);
            unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }.map_err(|e| cuda_err(OP, e))?;
        }
        Ok(self.wrap(out))
    }
}

/// Create a backend for CUDA device `ordinal` and register it for
/// `Device::Cuda(ordinal)`. Returns `Err` (never panics) when no CUDA driver,
/// runtime libraries, or device is present.
pub fn install(ordinal: u32) -> std::result::Result<(), String> {
    let backend = CudaBackend::new(ordinal)?;
    register_backend(Device::Cuda(ordinal), Arc::new(backend));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferro_core::dispatch::naive_matmul;
    use ferro_core::Tensor;

    // Compile-level proof that CudaBackend implements the full Backend trait
    // (including the *_dev overrides) and CudaBuf the DeviceBuffer trait.
    #[test]
    fn backend_trait_is_fully_implemented() {
        fn assert_backend<T: Backend>() {}
        fn assert_device_buffer<T: DeviceBuffer>() {}
        assert_backend::<CudaBackend>();
        assert_device_buffer::<CudaBuf>();
    }

    // Pure host reference for cublasSgemm column-major semantics with op():
    // C[i + j*ldc] = alpha * sum_p op(A)(i,p) * op(B)(p,j) + beta * C[..],
    // where op(X)(i,p) reads X[i + p*ld] under OP_N and X[p + i*ld] under OP_T.
    fn colmajor_sgemm(cfg: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
        let t = |op: cublasOperation_t| match op {
            cublasOperation_t::CUBLAS_OP_N => false,
            cublasOperation_t::CUBLAS_OP_T => true,
            other => panic!("unsupported op {other:?}"),
        };
        let (ta, tb) = (t(cfg.transa), t(cfg.transb));
        let (m, n, k) = (cfg.m as usize, cfg.n as usize, cfg.k as usize);
        let (lda, ldb, ldc) = (cfg.lda as usize, cfg.ldb as usize, cfg.ldc as usize);
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0;
                for p in 0..k {
                    let av = if ta { a[p + i * lda] } else { a[i + p * lda] };
                    let bv = if tb { b[j + p * ldb] } else { b[p + j * ldb] };
                    acc += av * bv;
                }
                c[i + j * ldc] = cfg.alpha * acc + cfg.beta * c[i + j * ldc];
            }
        }
    }

    // Validates the row-major -> column-major mapping without a GPU: applying
    // exact cuBLAS semantics to the swapped operands must reproduce the
    // row-major reference matmul, including output layout.
    #[test]
    fn sgemm_mapping_matches_row_major_matmul() {
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (2, 3, 4), (5, 2, 3), (1, 4, 2)] {
            let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5 - 1.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| 2.0 - i as f32 * 0.25).collect();
            let expected = naive_matmul(&a, &b, m, k, n);
            let cfg = row_major_sgemm_cfg(m, k, n, false, false);
            let mut c = vec![0.0f32; m * n];
            // Operand swap mirrors the gemm call: b is cuBLAS "A", a is "B".
            colmajor_sgemm(&cfg, &b, &a, &mut c);
            assert_eq!(c, expected, "mapping broken for ({m},{k},{n})");
        }
    }

    // Store a row-major (rows, cols) matrix as its (cols, rows) transpose.
    fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0f32; v.len()];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = v[i * cols + j];
            }
        }
        out
    }

    // The ta/tb flag mapping, all four combinations on non-square shapes: a
    // flagged operand is handed to sgemm in transposed storage, and simulated
    // cuBLAS semantics must still reproduce the logical (m,k)@(k,n) product.
    #[test]
    fn sgemm_transpose_flags_match_row_major_matmul() {
        for &(m, k, n) in &[(2usize, 3usize, 4usize), (5, 2, 3)] {
            let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5 - 1.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| 2.0 - i as f32 * 0.25).collect();
            let expected = naive_matmul(&a, &b, m, k, n);
            for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
                let sa = if ta { transpose(&a, m, k) } else { a.clone() };
                let sb = if tb { transpose(&b, k, n) } else { b.clone() };
                let cfg = row_major_sgemm_cfg(m, k, n, ta, tb);
                let mut c = vec![0.0f32; m * n];
                colmajor_sgemm(&cfg, &sb, &sa, &mut c);
                assert_eq!(c, expected, "mapping broken for ({m},{k},{n}) ta={ta} tb={tb}");
            }
        }
    }

    #[test]
    fn install_never_panics_without_gpu() {
        if is_available() {
            return; // exercised by gpu_end_to_end on GPU boxes
        }
        let res = install(0);
        assert!(res.is_err(), "install must fail without a CUDA driver");
        assert!(res.unwrap_err().contains("libcuda"));
        assert!(CudaBackend::new(0).is_err());
    }

    #[test]
    fn availability_probe_is_callable() {
        // Must not panic either way; false is the expected value on CI boxes
        // without a driver, and install() must then agree with it.
        if !is_available() {
            assert!(install(0).is_err());
        }
    }

    // Real-GPU smoke test: no-op without a driver, and tolerates a driver
    // with zero devices. On a GPU box it validates both the host-slice
    // fallback and the device-resident path end to end.
    #[test]
    fn gpu_end_to_end() {
        if !is_available() {
            return;
        }
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(_) => return, // driver present but no usable device
        };

        // Host-slice fallback path.
        let x = vec![-2.0f32, -0.5, 0.0, 1.5, 3.0];
        assert_eq!(backend.unary(UnaryKind::Relu, &x), vec![0.0, 0.0, 0.0, 1.5, 3.0]);
        assert_eq!(backend.unary(UnaryKind::Neg, &x), vec![2.0, 0.5, 0.0, -1.5, -3.0]);
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        assert_eq!(backend.binary(BinaryKind::Add, &a, &b), vec![11.0, 22.0, 33.0, 44.0]);
        let (m, k, n) = (2, 3, 2);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 + 1.0).collect();
        assert_eq!(backend.matmul(&a, &b, m, k, n), naive_matmul(&a, &b, m, k, n));

        // Direct *_dev round trip plus foreign-buffer rejection.
        let xd = backend.alloc_from_host(&x).unwrap();
        assert_eq!(xd.device(), Device::Cuda(0));
        assert_eq!(xd.len(), x.len());
        let rd = backend.unary_dev(UnaryKind::Relu, xd.as_ref()).unwrap();
        assert_eq!(backend.copy_to_host(rd.as_ref()).unwrap(), vec![0.0, 0.0, 0.0, 1.5, 3.0]);
        struct NotCuda;
        impl DeviceBuffer for NotCuda {
            fn device(&self) -> Device {
                Device::Cuda(0)
            }
            fn len(&self) -> usize {
                0
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        assert!(backend.unary_dev(UnaryKind::Relu, &NotCuda).is_err());

        // Resident tensor chain through core's dispatcher, mirroring
        // ferro-core/tests/device.rs: data stays on the GPU between ops.
        let dev = Device::Cuda(0);
        register_backend(dev, backend);
        let x = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]).unwrap();
        let w = Tensor::from_vec(vec![1.0, 0.5, -0.5, 1.0], &[2, 2]).unwrap();
        let xd = x.to_device(dev).unwrap();
        let wd = w.to_device(dev).unwrap();
        let out = xd.relu().exp().mul(&wd).unwrap().matmul(&wd).unwrap();
        assert_eq!(out.device(), dev);
        let host = out.to_device(Device::Cpu).unwrap().to_vec();
        let cpu = x.relu().exp().mul(&w).unwrap().matmul(&w).unwrap().to_vec();
        for (d, c) in host.iter().zip(cpu.iter()) {
            assert!((d - c).abs() < 1e-5, "device {d} vs cpu {c}");
        }

        // Resident training loop, mirroring ferro-core/tests/device.rs::
        // training_loop_stays_resident: 40 SGD steps of linear regression
        // (matmul + broadcast bias add + MSE + backward + tensor-op update)
        // run fully on Device::Cuda(0), converging and matching the cpu loop.
        let x = Tensor::from_vec(vec![1.0, 0.5, -0.3, 1.2, 0.7, -0.8, -1.1, 0.4], &[4, 2]).unwrap();
        let w_true = Tensor::from_vec(vec![2.0, -1.0], &[2, 1]).unwrap();
        let y = x.matmul(&w_true).unwrap().add(&Tensor::from_vec(vec![0.5], &[1]).unwrap()).unwrap();

        let run = |device: Option<Device>| -> (Vec<f32>, Vec<f32>, f32, f32) {
            let place = |t: Tensor| match device {
                Some(d) => t.to_device(d).unwrap(),
                None => t,
            };
            let xd = place(x.clone());
            let yd = place(y.clone());
            let lr = place(Tensor::scalar(0.1));
            let mut w = place(Tensor::from_vec(vec![0.0, 0.0], &[2, 1]).unwrap()).requires_grad_(true);
            let mut b = place(Tensor::from_vec(vec![0.0], &[1]).unwrap()).requires_grad_(true);
            let (mut first, mut last) = (f32::NAN, f32::NAN);
            for step in 0..40 {
                let pred = xd.matmul(&w).unwrap().add(&b).unwrap();
                let diff = pred.sub(&yd).unwrap();
                let loss = diff.mul(&diff).unwrap().mean();
                loss.backward();
                let (gw, gb) = (w.grad().unwrap(), b.grad().unwrap());
                if let Some(d) = device {
                    assert_eq!(gw.device(), d);
                    assert_eq!(gb.device(), d);
                }
                w = w.detach_copy().sub(&gw.mul(&lr).unwrap()).unwrap().requires_grad_(true);
                b = b.detach_copy().sub(&gb.mul(&lr).unwrap()).unwrap().requires_grad_(true);
                let l = loss.item();
                if step == 0 {
                    first = l;
                }
                last = l;
            }
            if let Some(d) = device {
                assert_eq!(w.device(), d);
                assert_eq!(b.device(), d);
            }
            (w.to_vec(), b.to_vec(), first, last)
        };

        let (w_dev, b_dev, first, last) = run(Some(dev));
        assert!(last < first * 0.05, "loss did not converge on device: {first} -> {last}");
        let (w_cpu, b_cpu, _, _) = run(None);
        for (d, c) in w_dev.iter().zip(w_cpu.iter()) {
            assert!((d - c).abs() < 1e-4, "w: device {d} vs cpu {c}");
        }
        for (d, c) in b_dev.iter().zip(b_cpu.iter()) {
            assert!((d - c).abs() < 1e-4, "b: device {d} vs cpu {c}");
        }
    }
}
