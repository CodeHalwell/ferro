//! CUDA backend for ferro, built on `cudarc` with runtime library loading.
//!
//! Storage is device-resident (dispatcher phase 3): the `*_dev` methods take
//! and return [`CudaBuf`]s wrapping `CudaSlice<f32>`, so chained ops keep
//! their data in GPU memory. The host-slice `Backend` methods remain as the
//! fallback path core uses for non-resident tensors; they are thin wrappers
//! (htod, `*_dev` compute, dtoh) over the same kernels.
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
use ferro_core::dispatch::DeviceBuffer;
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

    /// Row-major (m,k) @ (k,n) -> (m,n) over device-resident operands.
    fn sgemm(&self, op: &'static str, a: &CudaSlice<f32>, b: &CudaSlice<f32>, m: usize, k: usize, n: usize) -> Result<CudaSlice<f32>> {
        let mut c = self.stream.alloc_zeros::<f32>(m * n).map_err(|e| cuda_err(op, e))?;
        if k > 0 {
            let cfg = row_major_sgemm_cfg(m, k, n);
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
        let c = self.sgemm("matmul", &ad, &bd, m, k, n).expect("cublas sgemm failed");
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

    fn matmul_dev(&self, a: &dyn DeviceBuffer, b: &dyn DeviceBuffer, m: usize, k: usize, n: usize) -> Result<Box<dyn DeviceBuffer>> {
        let a = self.resident("matmul_dev", a)?;
        let b = self.resident("matmul_dev", b)?;
        Ok(self.wrap(self.sgemm("matmul_dev", &a.data, &b.data, m, k, n)?))
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
    }
}
