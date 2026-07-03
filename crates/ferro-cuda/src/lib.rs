//! CUDA backend for ferro, built on `cudarc` with runtime library loading.
//!
//! Storage is host-resident until dispatcher phase 3, so every op here is a
//! stopgap round trip: copy host -> device, compute, copy device -> host.
//! That is the honest contract of the current `Backend` seam (host slices in,
//! `Vec<f32>` out); tensors cannot yet live on the GPU.
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
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use ferro_core::{register_backend, Backend, BinaryKind, Device, UnaryKind};

/// Cheap detection: is the CUDA driver library (libcuda) loadable? True does
/// not guarantee a usable device (the driver can be present with zero GPUs);
/// [`install`] performs the real device init and reports failures as `Err`.
pub fn is_available() -> bool {
    unsafe { cudarc::driver::sys::is_culib_present() }
}

/// `Backend` implementation that stages host buffers through device memory.
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    // nvrtc-compiled elementwise kernels, keyed by generated source text so
    // parametrized kinds (Powf, Clamp) cache per scalar value.
    funcs: Mutex<HashMap<String, CudaFunction>>,
}

impl CudaBackend {
    /// Initialize device `ordinal`. Errors (rather than panicking) when the
    /// CUDA libraries are missing or the device cannot be initialized.
    pub fn new(ordinal: u32) -> Result<CudaBackend, String> {
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
        Ok(CudaBackend { ctx, stream, blas, funcs: Mutex::new(HashMap::new()) })
    }

    /// Fetch the cached kernel for `src`, compiling it with nvrtc on first
    /// use. Panics on failure: `Backend` methods have no error channel, and
    /// by this point a working device context exists, so failures are bugs
    /// (bad generated source) rather than environment issues.
    fn get_kernel(&self, src: &str) -> CudaFunction {
        let mut cache = self.funcs.lock().unwrap();
        if let Some(f) = cache.get(src) {
            return f.clone();
        }
        let ptx = compile_ptx(src).unwrap_or_else(|e| panic!("nvrtc failed to compile kernel: {e:?}\nsource:\n{src}"));
        let module = self.ctx.load_module(ptx).expect("failed to load compiled PTX module");
        let f = module.load_function(kernels::KERNEL_NAME).expect("compiled module is missing the kernel");
        cache.insert(src.to_string(), f.clone());
        f
    }

    fn launch_elementwise(&self, src: &str, inputs: &[&[f32]], n: usize) -> Vec<f32> {
        let func = self.get_kernel(src);
        let dev_inputs: Vec<_> = inputs
            .iter()
            .map(|h| self.stream.clone_htod(*h).expect("htod copy failed"))
            .collect();
        let mut out = self.stream.alloc_zeros::<f32>(n).expect("device alloc failed");
        let n_arg = n as u32;
        let mut launch = self.stream.launch_builder(&func);
        for d in &dev_inputs {
            launch.arg(d);
        }
        launch.arg(&mut out);
        launch.arg(&n_arg);
        unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }.expect("kernel launch failed");
        self.stream.clone_dtoh(&out).expect("dtoh copy failed")
    }
}

/// cuBLAS gemm parameters computing row-major C(m,n) = A(m,k) * B(k,n).
///
/// cuBLAS is column-major, so a row-major buffer of shape (r, c) is, viewed
/// column-major, the transposed (c, r) matrix. Rather than transposing, use
/// the identity C^T = B^T * A^T: a plain N/N sgemm over the column-major
/// views with the operands swapped (B as cuBLAS "A", A as cuBLAS "B") and
/// m/n swapped produces C^T column-major, whose bytes are exactly C
/// row-major. Leading dims are the row-major row strides: n for B and C,
/// k for A.
fn row_major_sgemm_cfg(m: usize, k: usize, n: usize) -> GemmConfig<f32> {
    GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0,
        ldc: n as i32,
    }
}

impl Backend for CudaBackend {
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }
        self.launch_elementwise(&kernels::unary_source(kind), &[x], x.len())
    }

    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.is_empty() {
            return Vec::new();
        }
        self.launch_elementwise(&kernels::binary_source(kind), &[a, b], a.len())
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        if m == 0 || n == 0 || k == 0 {
            return vec![0.0; m * n];
        }
        let a_dev = self.stream.clone_htod(a).expect("htod copy failed");
        let b_dev = self.stream.clone_htod(b).expect("htod copy failed");
        let mut c_dev = self.stream.alloc_zeros::<f32>(m * n).expect("device alloc failed");
        let cfg = row_major_sgemm_cfg(m, k, n);
        // Swapped operands: b_dev is cuBLAS "A", a_dev is cuBLAS "B".
        unsafe { self.blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev) }.expect("cublas sgemm failed");
        self.stream.clone_dtoh(&c_dev).expect("dtoh copy failed")
    }
}

/// Create a backend for CUDA device `ordinal` and register it for
/// `Device::Cuda(ordinal)`. Returns `Err` (never panics) when no CUDA driver,
/// runtime libraries, or device is present.
pub fn install(ordinal: u32) -> Result<(), String> {
    let backend = CudaBackend::new(ordinal)?;
    register_backend(Device::Cuda(ordinal), Arc::new(backend));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferro_core::dispatch::naive_matmul;

    // Pure host reference for cublasSgemm(N, N) column-major semantics:
    // C[i + j*ldc] = alpha * sum_p A[i + p*lda] * B[p + j*ldb] + beta * C[..]
    fn colmajor_sgemm(cfg: &GemmConfig<f32>, a: &[f32], b: &[f32], c: &mut [f32]) {
        assert_eq!(cfg.transa, cublasOperation_t::CUBLAS_OP_N);
        assert_eq!(cfg.transb, cublasOperation_t::CUBLAS_OP_N);
        let (m, n, k) = (cfg.m as usize, cfg.n as usize, cfg.k as usize);
        let (lda, ldb, ldc) = (cfg.lda as usize, cfg.ldb as usize, cfg.ldc as usize);
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[i + p * lda] * b[p + j * ldb];
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
            let cfg = row_major_sgemm_cfg(m, k, n);
            let mut c = vec![0.0f32; m * n];
            // Operand swap mirrors the gemm call: b is cuBLAS "A", a is "B".
            colmajor_sgemm(&cfg, &b, &a, &mut c);
            assert_eq!(c, expected, "mapping broken for ({m},{k},{n})");
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
    // with zero devices. On a GPU box it validates the full round trip.
    #[test]
    fn gpu_end_to_end() {
        if !is_available() {
            return;
        }
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => return, // driver present but no usable device
        };
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
    }
}
