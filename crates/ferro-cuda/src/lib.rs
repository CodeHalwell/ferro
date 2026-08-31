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
mod alloc;

pub use kernels::{chain_bc_args, chain_source, broadcast_strides, ChainStep};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use ferro_core::dispatch::{AdamWStep, DeviceBuffer, ReduceKind};
use ferro_core::{
    register_backend, Backend, BinaryKind, ChainStepRef, Device, Error, Result, UnaryKind,
};

/// Cheap detection: is the CUDA driver library (libcuda) loadable? True does
/// not guarantee a usable device (the driver can be present with zero GPUs);
/// [`install`] performs the real device init and reports failures as `Err`.
pub fn is_available() -> bool {
    unsafe { cudarc::driver::sys::is_culib_present() }
}

/// Map a cudarc failure into core's error type. The `*_dev` seam has a real
/// error channel, so driver errors surface as `Err` rather than panics.
fn cuda_err(op: &'static str, e: impl std::fmt::Display) -> Error {
    Error::Unsupported {
        op,
        msg: format!("CUDA error: {e}"),
    }
}

/// Kernel element counts are passed as `unsigned int` kernel arguments, so a
/// larger buffer cannot be launched; report it instead of truncating.
fn as_u32(op: &'static str, n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::Unsupported {
        op,
        msg: format!(
            "{n} elements exceeds the u32 kernel argument limit ({})",
            u32::MAX
        ),
    })
}

/// cuBLAS takes i32 leading dimensions and extents.
fn as_i32(op: &'static str, n: usize) -> Result<i32> {
    i32::try_from(n).map_err(|_| Error::Unsupported {
        op,
        msg: format!("{n} exceeds the cuBLAS i32 dimension limit"),
    })
}

/// Device-resident buffer: a `CudaSlice<f32>` tagged with the `Device` it
/// lives on. Core hands these back through `&dyn DeviceBuffer`; the backend
/// recovers the slice by downcasting via `as_any`.
pub struct CudaBuf {
    // ManuallyDrop so `Drop` can move the slice out and hand it to the
    // caching allocator for recycling instead of freeing it. Deref keeps
    // every `buf.data.len()` / method call site working unchanged; only
    // `&buf.data` reference sites need `&*buf.data`.
    data: std::mem::ManuallyDrop<CudaSlice<f32>>,
    device: Device,
    // Allocator this buffer's slice returns to on drop. Cheap Arc clone.
    alloc: crate::alloc::CachingAllocator,
}

impl std::ops::Deref for CudaBuf {
    type Target = CudaSlice<f32>;
    fn deref(&self) -> &CudaSlice<f32> {
        &self.data
    }
}

impl Drop for CudaBuf {
    fn drop(&mut self) {
        // SAFETY: `data` is never touched again after this take (we are in
        // drop), so the ManuallyDrop is not double-read.
        let slice = unsafe { std::mem::ManuallyDrop::take(&mut self.data) };
        self.alloc.recycle(slice);
    }
}

/// A whole training step (forward + backward + optimizer) recorded as one
/// CUDA graph. `replay()` re-executes every kernel with a single host launch.
///
/// Correctness rests on two coupled guarantees:
///  1. During capture the allocator BYPASSES its freelist, so every
///     intermediate allocation records a real graph alloc node and every drop
///     records a matching free node. Balanced alloc/free nodes make the graph
///     self-contained, so it can be instantiated `AUTO_FREE_ON_LAUNCH` and
///     safely relaunched — serving a recycled block from the freelist instead
///     would record a free node with no alloc node and fail replay with
///     CUDA_ERROR_INVALID_VALUE.
///  2. Params/grads/optimizer-state live OUTSIDE the capture window (allocated
///     before `begin_step_capture`), so they are never graph-managed; the
///     wave-6 in-place kernels mutate them at fixed addresses and they persist
///     and advance across replays. That is what makes replay a real step.
///
/// `_retained` normally ends up empty (intermediates free during capture); it
/// stays as a safety net for any slice the allocator chooses to pin, and is
/// dropped with the graph.
pub struct CapturedStep {
    _graph: cudarc::driver::safe::CudaGraph,
    _retained: Vec<CudaSlice<f32>>,
}

impl CapturedStep {
    /// Re-execute the entire captured step with one host call.
    pub fn replay(&self) -> Result<()> {
        self._graph
            .launch()
            .map_err(|e| cuda_err("captured_step_replay", e))
    }

    /// Number of device buffers the graph pins for its lifetime.
    pub fn retained_buffers(&self) -> usize {
        self._retained.len()
    }
}

/// A CUDA-graph-captured chain launch. `replay()` re-runs the recorded
/// launch with one host call, writing into the SAME output buffer every time
/// (stable-address contract). The graph and the output slice share this
/// struct's lifetime; dropping it destroys both.
pub struct CapturedChain {
    _graph: cudarc::driver::safe::CudaGraph,
    out: CudaSlice<f32>,
    buf: CudaBuf,
    n: usize,
}

impl CapturedChain {
    /// Replay the captured launch. Results appear in `output()` afterwards.
    pub fn replay(&self) -> Result<()> {
        self._graph.launch().map_err(|e| cuda_err("graph_replay", e))
    }
    /// The pre-allocated output buffer (last replay's results).
    pub fn output(&self) -> &CudaSlice<f32> {
        &self.out
    }
    /// Download the last replay's results to the host. Reads `self.out`
    /// directly - NOT the `buf` wrapper, whose data is a detached copy.
    pub fn copy_output_to_host(&self, backend: &CudaBackend) -> Result<Vec<f32>> {
        backend.dtoh("captured_output", &self.out)
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl CudaBackend {
    /// Run `f` with a mutable alias of `b`'s device allocation.
    /// `upgrade_device_ptr` re-wraps the SAME pointer (`CudaSlice::clone`
    /// would allocate and dtod-copy - mutating a clone mutates nothing), and
    /// `leak()` releases the alias without freeing on success and error
    /// alike, since dropping it would free the buffer out from under its
    /// real owner.
    fn with_alias_mut<R>(
        &self,
        b: &CudaBuf,
        f: impl FnOnce(&mut CudaSlice<f32>) -> Result<R>,
    ) -> Result<R> {
        let stream = b.data.stream();
        let (ptr, _sync) = DevicePtr::device_ptr(&*b.data, stream);
        // SAFETY: ptr/len come from a live CudaSlice on this stream's
        // context; the alias never outlives this call and is leaked, so
        // ownership stays with `b`.
        let mut alias: CudaSlice<f32> = unsafe { stream.upgrade_device_ptr(ptr, b.data.len()) };
        let res = f(&mut alias);
        let _ = alias.leak();
        res
    }

    /// Overwrite a device buffer's contents in place, preserving its address
    /// (the stable-address contract captured graphs and in-place ops rely
    /// on). Root-caused fix over the original: this used to `memcpy_htod`
    /// into `b.data.clone()`, but cudarc's `CudaSlice::clone` is
    /// `clone_dtod` - a fresh allocation - so the write landed in a
    /// temporary that was immediately dropped and the real buffer was never
    /// touched.
    pub fn copy_into(&self, buf: &dyn DeviceBuffer, data: &[f32]) -> Result<()> {
        const OP: &str = "copy_into";
        let b = self.resident(OP, buf)?;
        if b.data.len() != data.len() {
            return Err(Error::Unsupported {
                op: OP,
                msg: format!("length mismatch: {} vs {}", b.data.len(), data.len()),
            });
        }
        self.with_alias_mut(b, |alias| {
            self.stream
                .memcpy_htod(data, alias)
                .map_err(|e| cuda_err(OP, e))
        })
    }

    /// Launch an in-place elementwise kernel: `dst` is aliased mutably and
    /// passed first, then `extra` input buffers, then n and `scalars`, in
    /// the exact parameter order of the generated source.
    fn launch_inplace(
        &self,
        op: &'static str,
        src: &str,
        dst: &CudaBuf,
        extra: &[&CudaSlice<f32>],
        scalars: &[f32],
    ) -> Result<()> {
        let n = dst.data.len();
        if n == 0 {
            return Ok(());
        }
        let func = self.get_kernel(op, src)?;
        let n_arg = as_u32(op, n)?;
        self.with_alias_mut(dst, |alias| {
            let mut launch = self.stream.launch_builder(&func);
            launch.arg(alias);
            for e in extra {
                launch.arg(*e);
            }
            launch.arg(&n_arg);
            for s in scalars {
                launch.arg(s);
            }
            unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                .map(|_| ())
                .map_err(|e| cuda_err(op, e))
        })
    }
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

/// Device-resident i64 index buffer. A separate type (rather than a tag on
/// `CudaBuf`) so `resident`'s downcast rejects dtype mismatches structurally:
/// an f32 buffer can never be passed where an index buffer is expected.
pub struct CudaBufI64 {
    data: CudaSlice<i64>,
    device: Device,
}

impl DeviceBuffer for CudaBufI64 {
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
    // Ferro-owned caching allocator for f32 device buffers. Every CudaBuf
    // carries a clone and returns its slice here on drop.
    alloc: crate::alloc::CachingAllocator,
}

impl CudaBackend {
    /// Initialize device `ordinal`. Errors (rather than panicking) when the
    /// CUDA libraries are missing or the device cannot be initialized.
    pub fn new(ordinal: u32) -> std::result::Result<CudaBackend, String> {
        // cudarc's dynamic loader panics (not Err) on a missing library, so
        // probe all three libraries up front to keep this a clean Err path.
        unsafe {
            if !cudarc::driver::sys::is_culib_present() {
                return Err(
                    "CUDA driver library (libcuda) not found; no GPU driver installed".to_string(),
                );
            }
            if !cudarc::nvrtc::sys::is_culib_present() {
                return Err(
                    "NVRTC library (libnvrtc) not found; install the CUDA toolkit runtime"
                        .to_string(),
                );
            }
            if !cudarc::cublas::sys::is_culib_present() {
                return Err(
                    "cuBLAS library (libcublas) not found; install the CUDA toolkit runtime"
                        .to_string(),
                );
            }
        }
        let ctx = CudaContext::new(ordinal as usize)
            .map_err(|e| format!("failed to initialize CUDA device {ordinal}: {e}"))?;
        // A non-default stream: the legacy default stream is not capturable,
        // and graphs (wave 5) need a real stream anyway.
        let stream = ctx
            .new_stream()
            .map_err(|e| format!("failed to create stream: {e}"))?;
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("failed to create cuBLAS handle: {e}"))?;
        // cuBLAS allocates its workspace lazily from the memory pool; a
        // malloc inside stream capture silently invalidates the graph. Keep a
        // pre-allocated workspace so captured matmuls never allocate.
        {
            const BLAS_WS: usize = 4 << 20;
            let ws = unsafe { cudarc::driver::result::malloc_async(stream.cu_stream(), BLAS_WS) }
                .map_err(|e| format!("cublas workspace alloc failed: {e:?}"))?;
            unsafe {
                cudarc::cublas::sys::cublasSetWorkspace_v2(*blas.handle(), ws as _, BLAS_WS);
            }
            let _ = ws;
        }
        let device = Device::Cuda(ordinal);
        let alloc = crate::alloc::CachingAllocator::new(stream.clone());
        Ok(CudaBackend {
            ctx,
            stream,
            blas,
            device,
            funcs: Mutex::new(HashMap::new()),
            alloc,
        })
    }

    /// Build a backend whose allocator is a pure pass-through (no recycling):
    /// every buffer request hits the driver and every drop frees. This is the
    /// G6 baseline arm — identical code path, caching disabled — so the
    /// allocator's effect can be measured against it. Test/bench use only.
    #[doc(hidden)]
    pub fn new_passthrough(ordinal: u32) -> std::result::Result<CudaBackend, String> {
        let mut backend = CudaBackend::new(ordinal)?;
        backend.alloc = crate::alloc::CachingAllocator::passthrough(backend.stream.clone());
        Ok(backend)
    }

    /// Downcast a core-provided buffer back to this backend's `CudaBuf`,
    /// rejecting buffers from other backends or other CUDA devices.
    fn resident<'a>(&self, op: &'static str, buf: &'a dyn DeviceBuffer) -> Result<&'a CudaBuf> {
        let buf = buf
            .as_any()
            .downcast_ref::<CudaBuf>()
            .ok_or_else(|| Error::Unsupported {
                op,
                msg: "device buffer was not allocated by the CUDA backend".into(),
            })?;
        if buf.device != self.device {
            return Err(Error::Unsupported {
                op,
                msg: format!(
                    "buffer lives on {} but this backend serves {}",
                    buf.device, self.device
                ),
            });
        }
        Ok(buf)
    }

    fn wrap(&self, data: CudaSlice<f32>) -> Box<dyn DeviceBuffer> {
        Box::new(self.make_buf(data))
    }

    /// Build a `CudaBuf` carrying an allocator handle so its slice is recycled
    /// (not freed) on drop. Single construction point for f32 device buffers.
    fn make_buf(&self, data: CudaSlice<f32>) -> CudaBuf {
        CudaBuf {
            data: std::mem::ManuallyDrop::new(data),
            device: self.device,
            alloc: self.alloc.clone(),
        }
    }

    /// Allocate `len` zeroed f32s through the caching allocator (freelist hit
    /// when a same-size buffer was recycled, one driver call on a miss).
    fn alloc_zeros(&self, op: &'static str, len: usize) -> Result<CudaSlice<f32>> {
        self.alloc.alloc_zeros(len).map_err(|e| cuda_err(op, e))
    }

    /// Allocate `len` uninitialised f32s through the caching allocator. ONLY
    /// for buffers the caller fully overwrites before any read (matmul beta=0,
    /// elementwise maps, gather, fill). Skips the re-zero on a freelist hit,
    /// which is the allocator's fast path.
    ///
    /// # Safety
    /// Caller must write every element it later reads.
    unsafe fn alloc_uninit(&self, op: &'static str, len: usize) -> Result<CudaSlice<f32>> {
        self.alloc.alloc_uninit(len).map_err(|e| cuda_err(op, e))
    }

    /// Snapshot of the caching allocator's counters. Diff two snapshots across
    /// a training step for the G6 structural proof (zero fresh driver
    /// allocations per step after warm-up).
    pub fn alloc_stats(&self) -> crate::alloc::AllocStats {
        self.alloc.stats()
    }

    /// Number of slices currently held in the allocator's freelist.
    pub fn pooled_buffers(&self) -> usize {
        self.alloc.pooled()
    }

    /// Fetch the cached kernel for `src`, compiling it with nvrtc on first
    /// use. nvrtc failures on our generated source are bugs, but they are
    /// still reported as `Err` so callers never bring the process down.
    fn get_kernel(&self, op: &'static str, src: &str) -> Result<CudaFunction> {
        let mut cache = self.funcs.lock().unwrap();
        if let Some(f) = cache.get(src) {
            return Ok(f.clone());
        }
        let ptx = compile_ptx(src).map_err(|e| Error::Unsupported {
            op,
            msg: format!("nvrtc failed to compile kernel: {e}\nsource:\n{src}"),
        })?;
        let module = self.ctx.load_module(ptx).map_err(|e| cuda_err(op, e))?;
        let f = module
            .load_function(kernels::KERNEL_NAME)
            .map_err(|e| cuda_err(op, e))?;
        cache.insert(src.to_string(), f.clone());
        Ok(f)
    }

    /// Launch an elementwise kernel over device-resident inputs, writing to a
    /// freshly allocated output slice. No host round trip.
    fn launch_elementwise(
        &self,
        op: &'static str,
        src: &str,
        inputs: &[&CudaSlice<f32>],
        n: usize,
    ) -> Result<CudaSlice<f32>> {
        let func = self.get_kernel(op, src)?;
        let mut out = unsafe { self.alloc_uninit(op, n)? };
        // Zero-sized launches are invalid; an empty tensor's result is the
        // freshly allocated empty buffer.
        if n == 0 {
            return Ok(out);
        }
        let n_arg = as_u32(op, n)?;
        let mut launch = self.stream.launch_builder(&func);
        for d in inputs {
            launch.arg(*d);
        }
        launch.arg(&mut out);
        launch.arg(&n_arg);
        unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
            .map_err(|e| cuda_err(op, e))?;
        Ok(out)
    }

    /// Logical row-major (m,k) @ (k,n) -> (m,n) over device-resident
    /// operands; `ta`/`tb` mark an operand as stored transposed.
    #[allow(clippy::too_many_arguments)] // mirrors the Backend::matmul_dev seam
    fn sgemm(
        &self,
        op: &'static str,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
    ) -> Result<CudaSlice<f32>> {
        // cuBLAS extents and leading dimensions are i32.
        as_i32(op, m.max(k).max(n))?;
        // k==0 is matmul over an empty reduction dim: result is all-zeros, and
        // the cuBLAS call below is skipped, so the buffer must be zeroed rather
        // than an uninitialised (possibly recycled) slice. Only take the
        // alloc_uninit fast path when the GEMM will actually write every cell.
        let mut c = if k > 0 {
            unsafe { self.alloc_uninit(op, m * n)? }
        } else {
            self.alloc_zeros(op, m * n)?
        };
        if k > 0 {
            let cfg = row_major_sgemm_cfg(m, k, n, ta, tb);
            // Swapped operands: b is cuBLAS "A", a is cuBLAS "B".
            unsafe { self.blas.gemm(cfg, b, a, &mut c) }.map_err(|e| cuda_err(op, e))?;
        }
        Ok(c)
    }

    /// Batched GEMM over contiguous per-batch slabs: one strided-batched
    /// cuBLAS call for the whole (batch,m,k) @ (batch,k,n) product. Operand
    /// swap matches `sgemm`.
    #[allow(clippy::too_many_arguments)] // mirrors the Backend::bmm_dev seam
    fn sgemm_batched(
        &self,
        op: &'static str,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
    ) -> Result<CudaSlice<f32>> {
        as_i32(op, batch.max(m.max(k.max(n))))?;
        // As in `sgemm`: when any dim is 0 the cuBLAS call is skipped, so the
        // result must be a zeroed buffer (empty-reduction matmul is all-zeros),
        // not a recycled uninitialised slice. Fast path only on a real GEMM.
        let full = k > 0 && batch > 0 && m > 0 && n > 0;
        let mut c = if full {
            unsafe { self.alloc_uninit(op, batch * m * n)? }
        } else {
            self.alloc_zeros(op, batch * m * n)?
        };
        if full {
            let cfg = row_major_sgemm_strided_cfg(batch, m, k, n, ta, tb);
            unsafe { self.blas.gemm_strided_batched(cfg, b, a, &mut c) }
                .map_err(|e| cuda_err(op, e))?;
        }
        Ok(c)
    }

    fn htod(&self, op: &'static str, data: &[f32]) -> Result<CudaSlice<f32>> {
        // Allocate through the cache (recycling host-upload buffers too) then
        // fill. memcpy_htod fully overwrites, so the uninitialised fast path is
        // safe; this also means repeated same-size uploads (per-step batches)
        // hit the freelist instead of a fresh driver allocation each time.
        let mut slice = unsafe { self.alloc_uninit(op, data.len())? };
        self.stream
            .memcpy_htod(data, &mut slice)
            .map_err(|e| cuda_err(op, e))?;
        Ok(slice)
    }

    fn dtoh(&self, op: &'static str, data: &CudaSlice<f32>) -> Result<Vec<f32>> {
        self.stream.clone_dtoh(data).map_err(|e| cuda_err(op, e))
    }

    fn htod_i64(&self, op: &'static str, data: &[i64]) -> Result<CudaSlice<i64>> {
        self.stream.clone_htod(data).map_err(|e| cuda_err(op, e))
    }

    /// Row-wise softmax/log_softmax over a device-resident rows x cols buffer:
    /// pass 1 block-reduces per-row max and exp-sum into a stats buffer, pass 2
    /// applies them elementwise. Two launches, zero host round trips.
    fn run_row_softmax(
        &self,
        op: &'static str,
        x: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        log: bool,
    ) -> Result<CudaSlice<f32>> {
        let n = rows * cols;
        if n == 0 || rows == 0 {
            return self.alloc_zeros(op, n);
        }
        as_u32(op, n)?;
        let (rows_arg, cols_arg) = (as_u32(op, rows)?, as_u32(op, cols)?);
        let stats = self.alloc_zeros(op, 2 * rows)?;
        let sfunc = self.get_kernel(op, &kernels::softmax_row_stats_source())?;
        let mut launch = self.stream.launch_builder(&sfunc);
        launch.arg(x);
        launch.arg(&stats);
        launch.arg(&cols_arg);
        let cfg = LaunchConfig {
            grid_dim: (rows_arg, 1, 1),
            block_dim: (kernels::REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(cfg) }.map_err(|e| cuda_err(op, e))?;
        let out = self.launch_elementwise_apply(
            op,
            &kernels::softmax_apply_source(log),
            &[x, &stats],
            n,
            cols_arg,
        )?;
        Ok(out)
    }

    /// `launch_elementwise` with one extra trailing scalar arg (the apply
    /// kernel's `cols`); kept local so the shared helper stays untouched.
    fn launch_elementwise_apply(
        &self,
        op: &'static str,
        src: &str,
        inputs: &[&CudaSlice<f32>],
        n: usize,
        extra: u32,
    ) -> Result<CudaSlice<f32>> {
        let func = self.get_kernel(op, src)?;
        let mut out = unsafe { self.alloc_uninit(op, n)? };
        if n == 0 {
            return Ok(out);
        }
        let n_arg = as_u32(op, n)?;
        let mut launch = self.stream.launch_builder(&func);
        for d in inputs {
            launch.arg(*d);
        }
        launch.arg(&mut out);
        launch.arg(&n_arg);
        launch.arg(&extra);
        unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
            .map_err(|e| cuda_err(op, e))?;
        Ok(out)
    }

    fn dtoh_i64(&self, op: &'static str, data: &CudaSlice<i64>) -> Result<Vec<i64>> {
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
    let flag = |t: bool| {
        if t {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        }
    };
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

/// Same mapping as `row_major_sgemm_cfg` plus per-batch strides. Batch slabs
/// are contiguous (batch*m*k etc.), so the strides are just the slab sizes.
fn row_major_sgemm_strided_cfg(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
) -> StridedBatchedConfig<f32> {
    StridedBatchedConfig {
        gemm: row_major_sgemm_cfg(m, k, n, ta, tb),
        batch_size: batch as i32,
        stride_a: (k * n) as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    }
}

impl CudaBackend {
    /// Fallible host-slice compute used by the `Backend` fallback impls below.
    pub fn unary_res(&self, kind: UnaryKind, x: &[f32]) -> Result<Vec<f32>> {
        if x.is_empty() {
            return Ok(Vec::new());
        }
        let xd = self.htod("unary", x)?;
        let out =
            self.launch_elementwise("unary", &kernels::unary_source(kind), &[&xd], x.len())?;
        self.dtoh("unary", &out)
    }

    pub fn binary_res(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.is_empty() {
            return Ok(Vec::new());
        }
        let ad = self.htod("binary", a)?;
        let bd = self.htod("binary", b)?;
        let out = self.launch_elementwise(
            "binary",
            &kernels::binary_source(kind),
            &[&ad, &bd],
            a.len(),
        )?;
        self.dtoh("binary", &out)
    }

    pub fn matmul_res(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let ad = self.htod("matmul", a)?;
        let bd = self.htod("matmul", b)?;
        let c = self.sgemm("matmul", &ad, &bd, m, k, n, false, false)?;
        self.dtoh("matmul", &c)
    }

    /// Device-resident gradient seeding: allocate a fresh `CudaBuf` of `len`
    /// elements all equal to `value` via the device-side fill kernel -- no
    /// host round trip. This is what WS-B's tensor.rs should call through the
    /// Backend trait seam; the proposed trait addition in ferro-core's
    /// dispatch::Backend is:
    ///
    /// ```text
    /// /// Allocate a device buffer of `len` elements all equal to `value`
    ///   /// (device-side fill, no host round trip); used to seed backward
    ///   /// gradient buffers residently.
    ///   fn seed_grad_dev(&self, len: usize, value: f32) -> Result<Box<dyn DeviceBuffer>> {
    ///       self.fill_dev(value, len)
    ///   }
    /// ```
    ///
    /// The default delegating to fill_dev keeps every existing backend valid.
    pub fn seed_grad_dev(&self, len: usize, value: f32) -> Result<Box<dyn DeviceBuffer>> {
        self.fill_dev(value, len)
    }

    /// Launch the fused pointwise chain `steps` over device-resident inputs.
    /// Input 0 seeds the chain; its length is n. Inputs referenced by
    /// `ChainStep::Binary` must have length n; `BinaryBc` operands are read
    /// through their broadcast offsets and may be shorter. One launch, one
    /// output allocation -- no intermediate global-memory round trips.
    fn launch_chain(
        &self,
        op: &'static str,
        steps: &[kernels::ChainStep],
        inputs: &[&CudaBuf],
    ) -> Result<CudaSlice<f32>> {
        for buf in inputs {
            if buf.device != self.device {
                return Err(Error::Unsupported {
                    op,
                    msg: format!(
                        "buffer lives on {} but this backend serves {}",
                        buf.device, self.device
                    ),
                });
            }
        }
        let n = inputs[0].data.len();
        for step in steps {
            if let kernels::ChainStep::Binary { other, .. } = step {
                let got = inputs[*other].data.len();
                if got != n {
                    return Err(Error::Unsupported {
                        op,
                        msg: format!("elementwise chain input {other} has length {got} but expected {n}"),
                    });
                }
            }
        }
        // The kernel signature declares x0..x{m-1} for every referenced slot;
        // pass those pointers positionally.
        let n_inputs = kernels::chain_input_count(steps);
        assert!(inputs.len() >= n_inputs, "chain references a missing input");
        let func = self.get_kernel(op, &kernels::chain_source(steps))?;
        let mut out = unsafe { self.alloc_uninit(op, n)? };
        if n > 0 {
            let n_arg = as_u32(op, n)?;
            let mut launch = self.stream.launch_builder(&func);
            for buf in &inputs[..n_inputs] {
                launch.arg(&*buf.data);
            }
            // Signature order: input pointers, per-bc dims/strides, out, n.
            let extra = kernels::chain_bc_args(steps);
            for v in &extra {
                launch.arg(v);
            }
            launch.arg(&mut out);
            launch.arg(&n_arg);
            unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                .map_err(|e| cuda_err(op, e))?;
        }
        Ok(out)
    }

    /// Apply a pointwise-chain fusion plan to device-resident tensors.
    /// Opt-in until core's `dispatch::Backend` gains a chain seam (see
    /// docs/FUSION_SEAM.md); reachable only through this backend directly.
    pub fn chain_dev(
        &self,
        steps: &[kernels::ChainStep],
        inputs: &[&dyn DeviceBuffer],
    ) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "chain_dev";
        assert!(!inputs.is_empty(), "a chain needs at least the seed input");
        let bufs: Vec<&CudaBuf> = inputs
            .iter()
            .map(|b| self.resident(OP, *b))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.wrap(self.launch_chain(OP, steps, &bufs)?))
    }

    /// Begin capturing a whole training step on this backend's stream. The
    /// caller runs exactly one full step (forward, backward, optimizer) inside
    /// the window, then calls `end_step_capture`. Every kernel and scalar
    /// upload the step uses must have run at least once already (warm up two
    /// eager steps first) so no lazy nvrtc compile, module load, or cuBLAS
    /// workspace malloc lands mid-capture and invalidates the graph.
    ///
    /// The allocator enters capture mode for the window: freelist bypassed
    /// (every intermediate gets a fresh, unique address) and drops retained
    /// rather than recycled (no captured address is reused or freed early).
    /// That address discipline is what makes replay advance state correctly.
    pub fn begin_step_capture(self: &Arc<Self>) -> Result<()> {
        const OP: &str = "begin_step_capture";
        self.alloc.begin_capture();
        unsafe { self.ctx.disable_event_tracking() };
        let pre = self.stream.capture_status().map_err(|e| {
            unsafe { self.ctx.enable_event_tracking() };
            self.alloc.end_capture();
            cuda_err("begin.status", e)
        })?;
        if pre != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
            unsafe { self.ctx.enable_event_tracking() };
            self.alloc.end_capture();
            return Err(Error::Unsupported {
                op: OP,
                msg: format!("stream still capturing: {pre:?}"),
            });
        }
        self.stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| {
                unsafe { self.ctx.enable_event_tracking() };
                self.alloc.end_capture();
                cuda_err(OP, e)
            })
    }

    /// End the step-capture window and instantiate the recorded graph. The
    /// returned `CapturedStep` owns every buffer allocated during the window
    /// (the addresses the graph references) and replays the entire step with
    /// one host launch. Instantiated WITHOUT auto-free-on-launch: buffer
    /// lifetime is managed here via the retained set, not by the graph.
    pub fn end_step_capture(self: &Arc<Self>) -> Result<CapturedStep> {
        const OP: &str = "end_step_capture";
        // AUTO_FREE_ON_LAUNCH: the captured step allocates its intermediates
        // inside the window and frees them (recorded as free nodes), so the
        // graph is self-contained. This flag frees the graph's memory nodes at
        // the start of each launch and re-allocates on replay, which is the
        // correct lifecycle for a per-step training graph. Params live OUTSIDE
        // the capture window (allocated before begin_capture), so they are
        // never graph-managed and persist/advance across replays.
        let res = self.stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        unsafe { self.ctx.enable_event_tracking() };
        let retained = self.alloc.end_capture();
        let graph = match res {
            Ok(Some(g)) => g,
            Ok(None) => {
                return Err(Error::Unsupported {
                    op: OP,
                    msg: "step capture produced an empty graph".into(),
                })
            }
            Err(e) => return Err(cuda_err(OP, e)),
        };
        // Upload the instantiated graph to the stream before first launch,
        // exactly as capture_chain does; skipping it makes launch() fail with
        // CUDA_ERROR_INVALID_VALUE on some drivers.
        graph.upload().map_err(|e| cuda_err(OP, e))?;
        Ok(CapturedStep {
            _graph: graph,
            _retained: retained,
        })
    }

    /// CUDA-graph capture of a fused chain launch, plus a replay method.
    ///
    /// `capture_chain` allocates the output buffer FIRST (outside capture),
    /// then records one chain-kernel launch into a `CudaGraph` on this
    /// backend's stream and instantiates it. Because the output buffer is
    /// pre-allocated, every replay writes to the SAME memory - so replay is
    /// only valid until the returned `CudaSlice` is dropped. This is the
    /// static-shape, stable-address contract wave 5's step-level graphs
    /// build on.
    pub fn capture_chain(
        self: &Arc<Self>,
        steps: &[kernels::ChainStep],
        inputs: &[&dyn DeviceBuffer],
    ) -> Result<CapturedChain> {
        const OP: &str = "capture_chain";
        assert!(!inputs.is_empty(), "a chain needs at least the seed input");
        let bufs: Vec<&CudaBuf> = inputs
            .iter()
            .map(|b| self.resident(OP, *b))
            .collect::<Result<Vec<_>>>()?;
        let n = bufs[0].data.len();
        // SAFETY: no other stream work is in flight (single-stream backend);
        // cudarc requires disabling its event tracking for slices created
        // during capture, so do that around the whole capture scope. The
        // output buffer is allocated with ASYNC malloc from the stream's
        // memory pool BEFORE begin_capture, which is legal and keeps the
        // address stable for replays.
        // Warm the kernel cache and synchronize BEFORE capture: nvrtc
        // compile, module load, event creation, and stream syncs are not
        // capturable operations.
        let func = self.get_kernel(OP, &kernels::chain_source(steps))?;
        self.stream.synchronize().map_err(|e| cuda_err(OP, e))?;
        // Event tracking must be OFF for the whole capture scope: cudarc's
        // arg()/launch() record events on slices when it is on, and event
        // records inside a captured region are illegal (capture isolation).
        unsafe { self.ctx.disable_event_tracking() };
        let mut out = self.stream.alloc_zeros::<f32>(n).map_err(|e| {
            unsafe { self.ctx.enable_event_tracking() };
            cuda_err(OP, e)
        })?;
        let captured = (|| -> Result<_> {
            self.stream
                .begin_capture(cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| {
                    cuda_err(OP, e)
                })?;
            let launched = (|| -> Result<()> {
                if n == 0 {
                    return Ok(());
                }
                let n_arg = as_u32(OP, n)?;
                // Raw launch: the safe builder's arg()/launch() do event
                // bookkeeping and context syncs that are not capturable. The
                // CudaFunction/CudaSlice internals are inaccessible, so use
                // the safe builder but with event tracking disabled - which
                // makes is_managing_stream_synchronization() false and strips
                // all event work from arg()/launch().
                let mut launch = self.stream.launch_builder(&func);
                for buf in &bufs[..kernels::chain_input_count(steps)] {
                    launch.arg(&*buf.data);
                }
                let extra = kernels::chain_bc_args(steps);
                for v in &extra {
                    launch.arg(v);
                }
                let mut out_view = out.as_view_mut();
                launch.arg(&mut out_view);
                launch.arg(&n_arg);
                unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                    .map_err(|e| {
                        cuda_err(OP, e)
                    })?;
                Ok(())
            })();
            match launched {
                Ok(()) => match self
                    .stream
                    .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                {
                    Ok(Some(graph)) => Ok(graph),
                    Ok(None) => {
                        Err(Error::Unsupported {
                            op: OP,
                            msg: "stream capture produced an empty graph".into(),
                        })
                    }
                    Err(e) => {
                        Err(cuda_err(OP, e))
                    }
                },
                Err(e) => {
                    // Abort the poisoned capture so the stream stays usable.
                    let _ = self.stream.capture_status();
                    Err(e)
                }
            }
        })();
        unsafe { self.ctx.enable_event_tracking() };
        let graph = captured?;
        graph.upload().map_err(|e| cuda_err(OP, e))?;

        Ok(CapturedChain {
            _graph: graph,
            buf: self.make_buf(out.clone()),
            out,
            n,
        })
    }

    /// Host-slice convenience wrapper: htod every input, run [`chain_dev`]
    /// logic, dtoh the result. Used by tests and callers without resident
    /// buffers.
    pub fn chain_res(&self, steps: &[kernels::ChainStep], inputs: &[&[f32]]) -> Result<Vec<f32>> {
        const OP: &str = "chain";
        assert!(!inputs.is_empty(), "a chain needs at least the seed input");
        let dev: Vec<CudaSlice<f32>> = inputs
            .iter()
            .map(|d| self.htod(OP, d))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<CudaBuf> = dev.iter().map(|d| self.make_buf(d.clone())).collect();
        let ptrs: Vec<&CudaBuf> = refs.iter().collect();
        let out = self.launch_chain(OP, steps, &ptrs)?;
        self.dtoh(OP, &out)
    }
    /// Core-owned chain steps -> this backend's kernel step description.
    fn convert_steps(steps: &[ChainStepRef]) -> Vec<kernels::ChainStep> {
        steps
            .iter()
            .map(|s| match s {
                ChainStepRef::Unary(kind) => kernels::ChainStep::Unary(*kind),
                ChainStepRef::Binary { kind, other } => kernels::ChainStep::Binary {
                    kind: *kind,
                    other: *other,
                },
                ChainStepRef::BinaryBc {
                    kind,
                    other,
                    dims,
                    strides,
                } => kernels::ChainStep::BinaryBc {
                    kind: *kind,
                    other: *other,
                    dims: dims.clone(),
                    strides: strides.clone(),
                },
            })
            .collect()
    }
}

impl Backend for CudaBackend {
    // Host-slice fallbacks: htod, run the same device kernels as the *_dev
    // path, dtoh. The Backend seam returns bare Vec<f32> (no error channel),
    // so a driver failure cannot be propagated here; the real error path is
    // the *_res methods above (and the *_dev methods). A failure degrades to
    // an empty result with a stderr diagnostic rather than a panic.
    fn unary(&self, kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        self.unary_res(kind, x).unwrap_or_else(|e| {
            eprintln!("ferro-cuda: unary host fallback failed: {e}");
            Vec::new()
        })
    }

    fn binary(&self, kind: BinaryKind, a: &[f32], b: &[f32]) -> Vec<f32> {
        self.binary_res(kind, a, b).unwrap_or_else(|e| {
            eprintln!("ferro-cuda: binary host fallback failed: {e}");
            Vec::new()
        })
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        self.matmul_res(a, b, m, k, n).unwrap_or_else(|e| {
            eprintln!("ferro-cuda: matmul host fallback failed: {e}");
            Vec::new()
        })
    }

    fn alloc_from_host(&self, data: &[f32]) -> Result<Box<dyn DeviceBuffer>> {
        Ok(self.wrap(self.htod("alloc_from_host", data)?))
    }

    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>> {
        self.dtoh("copy_to_host", &self.resident("copy_to_host", buf)?.data)
    }

    fn unary_dev(&self, kind: UnaryKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        let x = self.resident("unary_dev", x)?;
        let out = self.launch_elementwise(
            "unary_dev",
            &kernels::unary_source(kind),
            &[&x.data],
            x.data.len(),
        )?;
        Ok(self.wrap(out))
    }

    fn binary_dev(
        &self,
        kind: BinaryKind,
        a: &dyn DeviceBuffer,
        b: &dyn DeviceBuffer,
    ) -> Result<Box<dyn DeviceBuffer>> {
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

    fn matmul_dev(
        &self,
        a: &dyn DeviceBuffer,
        b: &dyn DeviceBuffer,
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
    ) -> Result<Box<dyn DeviceBuffer>> {
        let a = self.resident("matmul_dev", a)?;
        let b = self.resident("matmul_dev", b)?;
        Ok(self.wrap(self.sgemm("matmul_dev", &a.data, &b.data, m, k, n, ta, tb)?))
    }

    fn bmm_dev(
        &self,
        a: &dyn DeviceBuffer,
        b: &dyn DeviceBuffer,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
    ) -> Result<Box<dyn DeviceBuffer>> {
        let a = self.resident("bmm_dev", a)?;
        let b = self.resident("bmm_dev", b)?;
        Ok(self.wrap(self.sgemm_batched("bmm_dev", &a.data, &b.data, batch, m, k, n, ta, tb)?))
    }

    fn binary_bc_dev(
        &self,
        kind: BinaryKind,
        a: &dyn DeviceBuffer,
        sa: &[usize],
        b: &dyn DeviceBuffer,
        sb: &[usize],
        out_shape: &[usize],
    ) -> Result<Box<dyn DeviceBuffer>> {
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
        let mut out = unsafe { self.alloc_uninit(OP, n)? };
        if n > 0 {
            let n_arg = as_u32(OP, n)?;
            let mut launch = self.stream.launch_builder(&func);
            launch.arg(&*a.data);
            launch.arg(&*b.data);
            launch.arg(&mut out);
            launch.arg(&n_arg);
            for d in dims.iter().chain(stra.iter()).chain(strb.iter()) {
                launch.arg(d);
            }
            unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                .map_err(|e| cuda_err(OP, e))?;
        }
        Ok(self.wrap(out))
    }

    fn chain_dev(
        &self,
        steps: &[ChainStepRef],
        inputs: &[&dyn DeviceBuffer],
    ) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "chain_dev";
        let bufs: Vec<&CudaBuf> = inputs
            .iter()
            .map(|b| self.resident(OP, *b))
            .collect::<Result<Vec<_>>>()?;
        let ksteps = Self::convert_steps(steps);
        Ok(self.wrap(self.launch_chain(OP, &ksteps, &bufs)?))
    }

    fn reduce_dev(&self, kind: ReduceKind, x: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "reduce_dev";
        let x = self.resident(OP, x)?;
        let n = x.data.len();
        // Two-pass tree reduction (see kernels.rs): pass 1 writes one partial
        // sum per block, pass 2 reduces the partials in a single block.
        if n == 0 {
            // Empty-input Sum is 0; Mean is 0/0 = NaN, matching core/torch.
            return match kind {
                ReduceKind::Sum => Ok(self.wrap(
                    self.alloc_zeros(OP, 1)?,
                )),
                ReduceKind::Mean => Ok(self.wrap(self.htod(OP, &[f32::NAN])?)),
            };
        }
        as_u32(OP, n)?;
        let blocks = kernels::reduce_grid(n);
        let partials = self.alloc_zeros(OP, blocks as usize)?;
        let pfunc = self.get_kernel(OP, &kernels::reduce_partial_source())?;
        let mut launch = self.stream.launch_builder(&pfunc);
        launch.arg(&*x.data);
        launch.arg(&partials);
        let n_arg = n as u32;
        launch.arg(&n_arg);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (kernels::REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(cfg) }.map_err(|e| cuda_err(OP, e))?;
        let ffunc = self.get_kernel("reduce_finalize", &kernels::reduce_finalize_source(kind))?;
        let mut out = self.alloc_zeros(OP, 1)?;
        let mut launch = self.stream.launch_builder(&ffunc);
        launch.arg(&partials);
        launch.arg(&mut out);
        launch.arg(&blocks);
        let total = n as u32;
        launch.arg(&total);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (kernels::REDUCE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(cfg) }.map_err(|e| cuda_err(OP, e))?;
        Ok(self.wrap(out))
    }

    fn sum_dim_dev(
        &self,
        x: &dyn DeviceBuffer,
        shape: &[usize],
        dim: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "sum_dim_dev";
        let x = self.resident(OP, x)?;
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();
        let n = outer * inner;
        let func = self.get_kernel(OP, &kernels::sum_dim_source())?;
        let mut out = self.alloc_zeros(OP, n)?;
        if n > 0 {
            let (red, inner) = (as_u32(OP, shape[dim])?, as_u32(OP, inner)?);
            let n_arg = as_u32(OP, n)?;
            let mut launch = self.stream.launch_builder(&func);
            launch.arg(&*x.data);
            launch.arg(&mut out);
            launch.arg(&n_arg);
            launch.arg(&red);
            launch.arg(&inner);
            unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                .map_err(|e| cuda_err(OP, e))?;
        }
        Ok(self.wrap(out))
    }

    fn softmax_dev(
        &self,
        x: &dyn DeviceBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        let x = self.resident("softmax_dev", x)?;
        Ok(self.wrap(self.run_row_softmax("softmax_dev", &x.data, rows, cols, false)?))
    }

    fn log_softmax_dev(
        &self,
        x: &dyn DeviceBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        let x = self.resident("log_softmax_dev", x)?;
        Ok(self.wrap(self.run_row_softmax("log_softmax_dev", &x.data, rows, cols, true)?))
    }

    fn fill_dev(&self, value: f32, len: usize) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "fill_dev";
        let func = self.get_kernel(OP, &kernels::fill_source())?;
        let mut out = unsafe { self.alloc_uninit(OP, len)? };
        if len > 0 {
            let n_arg = as_u32(OP, len)?;
            let mut launch = self.stream.launch_builder(&func);
            launch.arg(&mut out);
            launch.arg(&n_arg);
            launch.arg(&value);
            unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                .map_err(|e| cuda_err(OP, e))?;
        }
        Ok(self.wrap(out))
    }

    fn write_dev_from_host(&self, dst: &dyn DeviceBuffer, data: &[f32]) -> Result<()> {
        self.copy_into(dst, data)
    }

    fn copy_dev(&self, src: &dyn DeviceBuffer) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "copy_dev";
        let s = self.resident(OP, src)?;
        // Allocate the destination through the caching allocator (counted +
        // freelist-served) and device-to-device copy into it, rather than
        // try_clone() which bypasses the allocator: that would make a fresh
        // driver allocation every call, pool copies copy_dev never reuses, and
        // make alloc_stats() under-count fresh allocations for the interval.
        // memcpy_dtod fully overwrites, so the uninitialised fast path is safe.
        let mut dst = unsafe { self.alloc_uninit(OP, s.data.len())? };
        self.stream
            .memcpy_dtod(&*s.data, &mut dst)
            .map_err(|e| cuda_err(OP, e))?;
        Ok(self.wrap(dst))
    }

    fn copy_into_dev(&self, dst: &dyn DeviceBuffer, src: &dyn DeviceBuffer) -> Result<()> {
        const OP: &str = "copy_into_dev";
        let d = self.resident(OP, dst)?;
        let s = self.resident(OP, src)?;
        if d.data.len() != s.data.len() {
            return Err(Error::Unsupported {
                op: OP,
                msg: format!("length mismatch: {} vs {}", d.data.len(), s.data.len()),
            });
        }
        if d.data.len() == 0 {
            return Ok(());
        }
        // Kernel-based copy (NOT memcpy): a raw dtod memcpy can execute
        // eagerly instead of being recorded during stream capture, which
        // silently breaks captured training steps that rely on copy_ to
        // write updates into stable param storage. A kernel launch is
        // always a recorded node.
        self.launch_inplace(OP, &kernels::copy_source(), d, &[&s.data], &[])
    }

    fn fill_inplace_dev(&self, dst: &dyn DeviceBuffer, value: f32) -> Result<()> {
        const OP: &str = "fill_inplace_dev";
        let d = self.resident(OP, dst)?;
        // fill_source's parameter order is (out, n, value): reuse it with
        // the destination aliased as out.
        self.launch_inplace(OP, &kernels::fill_source(), d, &[], &[value])
    }

    fn affine_inplace_dev(&self, dst: &dyn DeviceBuffer, mul: f32, add: f32) -> Result<()> {
        const OP: &str = "affine_inplace_dev";
        let d = self.resident(OP, dst)?;
        self.launch_inplace(OP, &kernels::affine_inplace_source(), d, &[], &[mul, add])
    }

    fn binary_inplace_dev(
        &self,
        kind: BinaryKind,
        dst: &dyn DeviceBuffer,
        src: &dyn DeviceBuffer,
    ) -> Result<()> {
        const OP: &str = "binary_inplace_dev";
        let d = self.resident(OP, dst)?;
        let s = self.resident(OP, src)?;
        if d.data.len() != s.data.len() {
            return Err(Error::Unsupported {
                op: OP,
                msg: format!("length mismatch: {} vs {}", d.data.len(), s.data.len()),
            });
        }
        self.launch_inplace(OP, &kernels::binary_inplace_source(kind), d, &[&s.data], &[])
    }

    fn axpy_inplace_dev(
        &self,
        alpha: f32,
        dst: &dyn DeviceBuffer,
        src: &dyn DeviceBuffer,
    ) -> Result<()> {
        const OP: &str = "axpy_inplace_dev";
        let d = self.resident(OP, dst)?;
        let s = self.resident(OP, src)?;
        if d.data.len() != s.data.len() {
            return Err(Error::Unsupported {
                op: OP,
                msg: format!("length mismatch: {} vs {}", d.data.len(), s.data.len()),
            });
        }
        self.launch_inplace(OP, &kernels::axpy_inplace_source(), d, &[&s.data], &[alpha])
    }

    fn sgd_step_dev(
        &self,
        p: &dyn DeviceBuffer,
        v: &dyn DeviceBuffer,
        g: &dyn DeviceBuffer,
        lr: f32,
        momentum: f32,
        nesterov: bool,
    ) -> Result<()> {
        const OP: &str = "sgd_step_dev";
        let (bp, bv, bg) = (
            self.resident(OP, p)?,
            self.resident(OP, v)?,
            self.resident(OP, g)?,
        );
        let n = bp.data.len();
        if bv.data.len() != n || bg.data.len() != n {
            return Err(Error::Unsupported {
                op: OP,
                msg: "sgd_step_dev buffers must be same-length".into(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let func = self.get_kernel(OP, &kernels::sgd_step_source())?;
        let n_arg = as_u32(OP, n)?;
        let nesterov_arg: i32 = nesterov as i32;
        self.with_alias_mut(bp, |ap| {
            self.with_alias_mut(bv, |av| {
                let mut launch = self.stream.launch_builder(&func);
                launch.arg(ap);
                launch.arg(av);
                launch.arg(&*bg.data);
                launch.arg(&n_arg);
                launch.arg(&lr);
                launch.arg(&momentum);
                launch.arg(&nesterov_arg);
                unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                    .map(|_| ())
                    .map_err(|e| cuda_err(OP, e))
            })
        })
    }

    fn adamw_step_dev(
        &self,
        p: &dyn DeviceBuffer,
        m: &dyn DeviceBuffer,
        v: &dyn DeviceBuffer,
        g: &dyn DeviceBuffer,
        hp: AdamWStep,
    ) -> Result<()> {
        const OP: &str = "adamw_step_dev";
        let (bp, bm, bv, bg) = (
            self.resident(OP, p)?,
            self.resident(OP, m)?,
            self.resident(OP, v)?,
            self.resident(OP, g)?,
        );
        let n = bp.data.len();
        if bm.data.len() != n || bv.data.len() != n || bg.data.len() != n {
            return Err(Error::Unsupported {
                op: OP,
                msg: "adamw_step_dev buffers must be same-length".into(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let func = self.get_kernel(OP, &kernels::adamw_step_source())?;
        let n_arg = as_u32(OP, n)?;
        self.with_alias_mut(bp, |ap| {
            self.with_alias_mut(bm, |am| {
                self.with_alias_mut(bv, |av| {
                    let mut launch = self.stream.launch_builder(&func);
                    launch.arg(ap);
                    launch.arg(am);
                    launch.arg(av);
                    launch.arg(&*bg.data);
                    launch.arg(&n_arg);
                    launch.arg(&hp.lr);
                    launch.arg(&hp.beta1);
                    launch.arg(&hp.beta2);
                    launch.arg(&hp.bc1);
                    launch.arg(&hp.bc2);
                    launch.arg(&hp.eps);
                    launch.arg(&hp.weight_decay);
                    unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                        .map(|_| ())
                        .map_err(|e| cuda_err(OP, e))
                })
            })
        })
    }

    fn alloc_i64_from_host(&self, data: &[i64]) -> Result<Box<dyn DeviceBuffer>> {
        Ok(Box::new(CudaBufI64 {
            data: self.htod_i64("alloc_i64_from_host", data)?,
            device: self.device,
        }))
    }

    fn copy_i64_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<i64>> {
        let buf = buf
            .as_any()
            .downcast_ref::<CudaBufI64>()
            .ok_or_else(|| Error::Unsupported {
                op: "copy_i64_to_host",
                msg: "buffer was not an i64 buffer allocated by the CUDA backend".into(),
            })?;
        if buf.device != self.device {
            return Err(Error::Unsupported {
                op: "copy_i64_to_host",
                msg: format!(
                    "buffer lives on {} but this backend serves {}",
                    buf.device, self.device
                ),
            });
        }
        self.dtoh_i64("copy_i64_to_host", &buf.data)
    }

    fn gather_rows_dev(
        &self,
        w: &dyn DeviceBuffer,
        idx: &dyn DeviceBuffer,
        dim_size: usize,
        inner: usize,
    ) -> Result<Box<dyn DeviceBuffer>> {
        const OP: &str = "gather_rows_dev";
        let w = self.resident(OP, w)?;
        let idx = idx
            .as_any()
            .downcast_ref::<CudaBufI64>()
            .ok_or_else(|| Error::Unsupported {
                op: OP,
                msg: "index buffer was not an i64 buffer allocated by the CUDA backend".into(),
            })?;
        if idx.device != self.device {
            return Err(Error::Unsupported {
                op: OP,
                msg: format!(
                    "buffer lives on {} but this backend serves {}",
                    idx.device, self.device
                ),
            });
        }
        // The kernel indexes w by idx[o]*inner + j with unsigned int math;
        // dim_size * inner must stay inside that range.
        as_u32(OP, inner)?;
        let rows = idx.data.len();
        let n = rows * inner;
        let func = self.get_kernel(OP, &kernels::gather_source())?;
        let mut out = unsafe { self.alloc_uninit(OP, n)? };
        if n > 0 {
            let (inner_arg, n_arg) = (as_u32(OP, inner)?, as_u32(OP, n)?);
            let _ = dim_size; // bounds were validated on the host before upload
            let mut launch = self.stream.launch_builder(&func);
            launch.arg(&idx.data);
            launch.arg(&*w.data);
            launch.arg(&mut out);
            launch.arg(&inner_arg);
            launch.arg(&n_arg);
            unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                .map_err(|e| cuda_err(OP, e))?;
        }
        Ok(self.wrap(out))
    }
}

impl CudaBackend {
    /// Capturable AdamW step: like `adamw_step_dev` but reads the timestep from
    /// a device-resident 1-element buffer `t` and computes bias correction
    /// in-kernel, so the step is safe to record in a CUDA graph and replay with
    /// an advancing correction. `t` must already hold this step's index (the
    /// caller bumps it via `scalar_increment_dev` first, mirroring the host
    /// path's `self.t += 1` before computing bc). `beta1/beta2/eps/wd/lr`
    /// remain host constants baked into the launch, matching PyTorch capturable
    /// Adam where only lr/step are allowed to vary across replays.
    pub(crate) fn adamw_step_capturable_dev(
        &self,
        p: &dyn DeviceBuffer,
        m: &dyn DeviceBuffer,
        v: &dyn DeviceBuffer,
        g: &dyn DeviceBuffer,
        t: &dyn DeviceBuffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Result<()> {
        const OP: &str = "adamw_step_capturable_dev";
        let (bp, bm, bv, bg, bt) = (
            self.resident(OP, p)?,
            self.resident(OP, m)?,
            self.resident(OP, v)?,
            self.resident(OP, g)?,
            self.resident(OP, t)?,
        );
        let n = bp.data.len();
        if bm.data.len() != n || bv.data.len() != n || bg.data.len() != n {
            return Err(Error::Unsupported {
                op: OP,
                msg: "adamw_step_capturable_dev buffers must be same-length".into(),
            });
        }
        if bt.data.len() != 1 {
            return Err(Error::Unsupported {
                op: OP,
                msg: "adamw_step_capturable_dev timestep buffer must have length 1".into(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let func = self.get_kernel(OP, &kernels::adamw_step_capturable_source())?;
        let n_arg = as_u32(OP, n)?;
        self.with_alias_mut(bp, |ap| {
            self.with_alias_mut(bm, |am| {
                self.with_alias_mut(bv, |av| {
                    let mut launch = self.stream.launch_builder(&func);
                    launch.arg(ap);
                    launch.arg(am);
                    launch.arg(av);
                    launch.arg(&*bg.data);
                    launch.arg(&*bt.data);
                    launch.arg(&n_arg);
                    launch.arg(&lr);
                    launch.arg(&beta1);
                    launch.arg(&beta2);
                    launch.arg(&eps);
                    launch.arg(&weight_decay);
                    unsafe { launch.launch(LaunchConfig::for_num_elems(n_arg)) }
                        .map(|_| ())
                        .map_err(|e| cuda_err(OP, e))
                })
            })
        })
    }

    /// Advance a device-resident scalar timestep by one (`t[0] += 1`). Runs as
    /// a one-thread kernel so it is captured as a graph node and re-executed on
    /// every replay.
    pub(crate) fn scalar_increment_dev(&self, t: &dyn DeviceBuffer) -> Result<()> {
        const OP: &str = "scalar_increment_dev";
        let bt = self.resident(OP, t)?;
        if bt.data.len() != 1 {
            return Err(Error::Unsupported {
                op: OP,
                msg: "scalar_increment_dev buffer must have length 1".into(),
            });
        }
        let func = self.get_kernel(OP, &kernels::scalar_increment_source())?;
        self.with_alias_mut(bt, |at| {
            let mut launch = self.stream.launch_builder(&func);
            launch.arg(at);
            unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                .map(|_| ())
                .map_err(|e| cuda_err(OP, e))
        })
    }
}

/// Create a backend for CUDA device `ordinal` and register it for
/// `Device::Cuda(ordinal)`. Returns `Err` (never panics) when no CUDA driver,
/// runtime libraries, or device is present.
pub fn install(ordinal: u32) -> std::result::Result<(), String> {
    let backend = Arc::new(CudaBackend::new(ordinal)?);
    // Record the Arc so `cuda_backend()` can hand capture/replay handles to
    // callers that own the step loop (benchmarks, examples) and need the
    // CudaBackend surface beyond the `Backend` trait. Ignore a second install.
    let _ = LAST_BACKEND.set(backend.clone());
    register_backend(Device::Cuda(ordinal), backend);
    Ok(())
}

/// The most recently installed CUDA backend, for callers (benchmarks,
/// examples) that own the training loop and need capture/replay handles
/// (`begin_step_capture`/`end_step_capture`) not exposed on the `Backend`
/// trait. `None` before `install` succeeds.
pub fn cuda_backend() -> Option<Arc<CudaBackend>> {
    LAST_BACKEND.get().cloned()
}

static LAST_BACKEND: std::sync::OnceLock<Arc<CudaBackend>> = std::sync::OnceLock::new();

/// Raw parts for zero-copy DLPack export of a `CudaBuf`: its base device
/// pointer and device ordinal. The caller must keep the buffer alive while
/// using the pointer (the DLPack capsule holds an Arc of the storage).
pub fn exported_view(buf: &dyn DeviceBuffer) -> std::result::Result<(usize, u32), String> {
    let b = buf
        .as_any()
        .downcast_ref::<CudaBuf>()
        .ok_or_else(|| "device buffer was not allocated by the CUDA backend".to_string())?;
    let ordinal = match b.device {
        Device::Cuda(n) => n,
        other => return Err(format!("unexpected device {other} on a CUDA buffer")),
    };
    let stream = b.data.stream();
    let (ptr, _sync) = DevicePtr::device_ptr(&*b.data, stream);
    Ok((ptr as usize, ordinal))
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
                assert_eq!(
                    c, expected,
                    "mapping broken for ({m},{k},{n}) ta={ta} tb={tb}"
                );
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
    fn strided_batched_cfg_matches_per_slab_single_gemms() {
        // CPU proof that the strided config maps each batch slab identically
        // to the single-GEMM mapping: per-slab colmajor_sgemm with
        // row_major_sgemm_cfg must equal a naive batched product with the
        // same logical transpose flags (the semantics sgemm_batched gets).
        fn naive_bmm(
            a: &[f32],
            b: &[f32],
            batch: usize,
            m: usize,
            k: usize,
            n: usize,
            ta: bool,
            tb: bool,
        ) -> Vec<f32> {
            let mut out = vec![0.0f32; batch * m * n];
            for bi in 0..batch {
                let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0f32;
                        for p in 0..k {
                            // Raw slabs are always logical row-major (m,k) and
                            // (k,n); the ta/tb flags only change how cuBLAS is
                            // told to read them, not the storage.
                            let av = a[ao + i * k + p];
                            let bv = b[bo + p * n + j];
                            acc += av * bv;
                        }
                        out[co + i * n + j] = acc;
                    }
                }
            }
            out
        }
        for &(batch, m, k, n) in &[(1usize, 1usize, 1usize, 1usize), (2, 2, 3, 2), (3, 1, 4, 2)] {
            for &(ta, tb) in &[(false, false), (false, true), (true, false)] {
                let a: Vec<f32> = (0..batch * m * k)
                    .map(|i| (i as f32 * 0.5 - 1.0).sin())
                    .collect();
                let b: Vec<f32> = (0..batch * k * n).map(|i| 2.0 - i as f32 * 0.25).collect();
                let mut per_slab = vec![0.0f32; batch * m * n];
                for bi in 0..batch {
                    let (ao, bo, co) = (bi * m * k, bi * k * n, bi * m * n);
                    let sa = if ta {
                        transpose(&a[ao..ao + m * k], m, k)
                    } else {
                        a[ao..ao + m * k].to_vec()
                    };
                    let sb = if tb {
                        transpose(&b[bo..bo + k * n], k, n)
                    } else {
                        b[bo..bo + k * n].to_vec()
                    };
                    let cfg = row_major_sgemm_cfg(m, k, n, ta, tb);
                    let mut c = vec![0.0f32; m * n];
                    colmajor_sgemm(&cfg, &sb, &sa, &mut c);
                    per_slab[co..co + m * n].copy_from_slice(&c);
                }
                let expected = naive_bmm(&a, &b, batch, m, k, n, ta, tb);
                assert_eq!(
                    per_slab, expected,
                    "strided mapping mismatch b={batch} ({m},{k},{n}) ta={ta} tb={tb}"
                );
            }
        }
    }

    // Real-GPU smoke test: no-op without a driver, and tolerates a driver
    // with zero devices. On a GPU box it validates both the host-slice
    // fallback and the device-resident path end to end.
    /// Gated like gpu_end_to_end: the in-place kernels must mutate the
    /// ORIGINAL allocation (copy_into once wrote into a dtod clone and was a
    /// silent no-op - this is its regression test), and the fused steps must
    /// match the host reference loops bitwise.
    #[test]
    fn gpu_inplace_kernels_mutate_the_original_buffer() {
        if !is_available() {
            return;
        }
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(_) => return,
        };
        let buf = backend.alloc_from_host(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        backend.copy_into(buf.as_ref(), &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(
            backend.copy_to_host(buf.as_ref()).unwrap(),
            vec![5.0, 6.0, 7.0, 8.0],
            "copy_into must write the original allocation, not a clone"
        );

        backend.fill_inplace_dev(buf.as_ref(), 2.0).unwrap();
        assert_eq!(backend.copy_to_host(buf.as_ref()).unwrap(), vec![2.0; 4]);

        backend.affine_inplace_dev(buf.as_ref(), 3.0, 1.0).unwrap();
        assert_eq!(backend.copy_to_host(buf.as_ref()).unwrap(), vec![7.0; 4]);

        let src = backend.alloc_from_host(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        backend
            .binary_inplace_dev(BinaryKind::Sub, buf.as_ref(), src.as_ref())
            .unwrap();
        assert_eq!(
            backend.copy_to_host(buf.as_ref()).unwrap(),
            vec![6.0, 5.0, 4.0, 3.0]
        );
        backend
            .axpy_inplace_dev(0.5, buf.as_ref(), src.as_ref())
            .unwrap();
        assert_eq!(
            backend.copy_to_host(buf.as_ref()).unwrap(),
            vec![6.5, 6.0, 5.5, 5.0]
        );
        // Aliased dst == src is the in-place self-op contract.
        backend
            .binary_inplace_dev(BinaryKind::Add, buf.as_ref(), buf.as_ref())
            .unwrap();
        assert_eq!(
            backend.copy_to_host(buf.as_ref()).unwrap(),
            vec![13.0, 12.0, 11.0, 10.0]
        );

        // Fused steps vs the trait's host reference loops: bitwise equal.
        let (p0, v0, g0) = (
            vec![1.0f32, -0.5, 0.25, 2.0],
            vec![0.1f32, 0.0, -0.2, 0.05],
            vec![0.3f32, -0.7, 0.9, -0.111],
        );
        let (pd, vd, gd) = (
            backend.alloc_from_host(&p0).unwrap(),
            backend.alloc_from_host(&v0).unwrap(),
            backend.alloc_from_host(&g0).unwrap(),
        );
        let (mut ph, mut vh) = (p0.clone(), v0.clone());
        for step in 0..3 {
            backend
                .sgd_step_dev(pd.as_ref(), vd.as_ref(), gd.as_ref(), 0.1, 0.9, step % 2 == 0)
                .unwrap();
            Backend::sgd_step(&*backend, &mut ph, &mut vh, &g0, 0.1, 0.9, step % 2 == 0);
        }
        assert_eq!(backend.copy_to_host(pd.as_ref()).unwrap(), ph);
        assert_eq!(backend.copy_to_host(vd.as_ref()).unwrap(), vh);

        let hp = ferro_core::dispatch::AdamWStep {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            bc1: 1.0 - 0.9f32.powi(1),
            bc2: 1.0 - 0.999f32.powi(1),
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let (m0, a0) = (vec![0.0f32; 4], p0.clone());
        let (pa, ma, va) = (
            backend.alloc_from_host(&a0).unwrap(),
            backend.alloc_from_host(&m0).unwrap(),
            backend.alloc_from_host(&m0).unwrap(),
        );
        let (mut pah, mut mah, mut vah) = (a0.clone(), m0.clone(), m0.clone());
        backend
            .adamw_step_dev(pa.as_ref(), ma.as_ref(), va.as_ref(), gd.as_ref(), hp)
            .unwrap();
        Backend::adamw_step(&*backend, &mut pah, &mut mah, &mut vah, &g0, hp);
        assert_eq!(backend.copy_to_host(pa.as_ref()).unwrap(), pah);
        assert_eq!(backend.copy_to_host(ma.as_ref()).unwrap(), mah);
        assert_eq!(backend.copy_to_host(va.as_ref()).unwrap(), vah);
    }

    /// The capturable AdamW path (device timestep + in-kernel bias correction)
    /// must match the host `adamw_step` reference across several steps. The
    /// kernel computes `1 - beta^t` with `__powf` while the host uses `powi`,
    /// so results agree to f32 tolerance rather than bit-identically.
    #[test]
    fn adamw_capturable_dev_matches_host_reference() {
        if !is_available() {
            return;
        }
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(_) => return,
        };
        let (lr, beta1, beta2, eps, wd) = (0.01f32, 0.9f32, 0.999f32, 1e-8f32, 0.01f32);
        let p0 = vec![0.5f32, -1.0, 2.0, -0.25];
        let g0 = vec![0.1f32, -0.2, 0.3, -0.4];
        let m0 = vec![0.0f32; 4];
        let (pd, md, vd, gd, td) = (
            backend.alloc_from_host(&p0).unwrap(),
            backend.alloc_from_host(&m0).unwrap(),
            backend.alloc_from_host(&m0).unwrap(),
            backend.alloc_from_host(&g0).unwrap(),
            backend.alloc_from_host(&[0.0f32]).unwrap(),
        );
        let (mut ph, mut mh, mut vh) = (p0.clone(), m0.clone(), m0.clone());
        for step in 1..=5i32 {
            // Device path: bump timestep on-device, then run capturable step.
            backend.scalar_increment_dev(td.as_ref()).unwrap();
            backend
                .adamw_step_capturable_dev(
                    pd.as_ref(),
                    md.as_ref(),
                    vd.as_ref(),
                    gd.as_ref(),
                    td.as_ref(),
                    lr,
                    beta1,
                    beta2,
                    eps,
                    wd,
                )
                .unwrap();
            // Host reference with the same step's host-computed bias correction.
            let hp = ferro_core::dispatch::AdamWStep {
                lr,
                beta1,
                beta2,
                bc1: 1.0 - beta1.powi(step),
                bc2: 1.0 - beta2.powi(step),
                eps,
                weight_decay: wd,
            };
            Backend::adamw_step(&*backend, &mut ph, &mut mh, &mut vh, &g0, hp);
        }
        // Confirm the device timestep advanced to 5.
        assert_eq!(backend.copy_to_host(td.as_ref()).unwrap(), vec![5.0]);
        let got_p = backend.copy_to_host(pd.as_ref()).unwrap();
        let got_m = backend.copy_to_host(md.as_ref()).unwrap();
        let got_v = backend.copy_to_host(vd.as_ref()).unwrap();
        for i in 0..4 {
            // 1e-5: __powf (device) vs powi (host) diverge by ~1e-6 per step and
            // accumulate over the 5 steps; the update math is otherwise identical.
            assert!(
                (got_p[i] - ph[i]).abs() < 1e-5,
                "param {i}: capturable {} vs host {}",
                got_p[i],
                ph[i]
            );
            assert!((got_m[i] - mh[i]).abs() < 1e-5, "m {i} mismatch");
            assert!((got_v[i] - vh[i]).abs() < 1e-5, "v {i} mismatch");
        }
    }

    /// PROOF that the capturable AdamW step survives CUDA-graph capture with an
    /// ADVANCING bias correction — the whole reason it exists. The device
    /// timestep `t`, params `p`, and moments `m`/`v` all live OUTSIDE the
    /// capture window; the graph records `t += 1` then the capturable step, so
    /// each replay bumps the timestep and applies that step's bias correction.
    /// The host-scalar `adamw_step_dev` would freeze bc at capture time and fail
    /// this test (every replay re-applying step-1's correction).
    #[test]
    fn captured_adamw_advances_bias_correction_across_replays() {
        if !is_available() {
            return;
        }
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(_) => return,
        };
        let (lr, beta1, beta2, eps, wd) = (0.01f32, 0.9f32, 0.999f32, 1e-8f32, 0.0f32);
        let p0 = vec![0.5f32, -1.0, 2.0, -0.25];
        let g0 = vec![0.1f32, -0.2, 0.3, -0.4];
        let z = vec![0.0f32; 4];

        // Eager reference trajectory over N capturable steps (host bc per step).
        let run_eager = |steps: usize| -> Vec<f32> {
            let (pd, md, vd, gd, td) = (
                backend.alloc_from_host(&p0).unwrap(),
                backend.alloc_from_host(&z).unwrap(),
                backend.alloc_from_host(&z).unwrap(),
                backend.alloc_from_host(&g0).unwrap(),
                backend.alloc_from_host(&[0.0f32]).unwrap(),
            );
            for _ in 0..steps {
                backend.scalar_increment_dev(td.as_ref()).unwrap();
                backend
                    .adamw_step_capturable_dev(
                        pd.as_ref(), md.as_ref(), vd.as_ref(), gd.as_ref(), td.as_ref(),
                        lr, beta1, beta2, eps, wd,
                    )
                    .unwrap();
            }
            backend.copy_to_host(pd.as_ref()).unwrap()
        };

        const N: usize = 6;
        let ref_final = run_eager(N);

        // Captured path: allocate state OUTSIDE the window, record one step.
        let (pd, md, vd, gd, td) = (
            backend.alloc_from_host(&p0).unwrap(),
            backend.alloc_from_host(&z).unwrap(),
            backend.alloc_from_host(&z).unwrap(),
            backend.alloc_from_host(&g0).unwrap(),
            backend.alloc_from_host(&[0.0f32]).unwrap(),
        );
        if let Err(e) = backend.begin_step_capture() {
            eprintln!("begin_step_capture unsupported: {e}; skipping");
            return;
        }
        backend.scalar_increment_dev(td.as_ref()).unwrap();
        backend
            .adamw_step_capturable_dev(
                pd.as_ref(), md.as_ref(), vd.as_ref(), gd.as_ref(), td.as_ref(),
                lr, beta1, beta2, eps, wd,
            )
            .unwrap();
        let step = match backend.end_step_capture() {
            Ok(s) => s,
            Err(e) => panic!("end_step_capture failed: {e}"),
        };

        // Capture RECORDS but does not execute: state is untouched, t still 0.
        assert_eq!(backend.copy_to_host(td.as_ref()).unwrap(), vec![0.0]);
        assert_eq!(backend.copy_to_host(pd.as_ref()).unwrap(), p0);

        // Replay N times; t must climb 1..=N and params track the eager run.
        for s in 1..=N {
            step.replay().expect("replay");
            assert_eq!(
                backend.copy_to_host(td.as_ref()).unwrap(),
                vec![s as f32],
                "timestep must advance each replay (frozen bc bug freezes it)"
            );
        }
        let got_final = backend.copy_to_host(pd.as_ref()).unwrap();
        for i in 0..4 {
            assert!(
                (got_final[i] - ref_final[i]).abs() < 1e-5,
                "param {i}: replayed {} vs eager {} — bias correction did not advance under replay",
                got_final[i],
                ref_final[i]
            );
        }
    }

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
        assert_eq!(
            backend.unary(UnaryKind::Relu, &x),
            vec![0.0, 0.0, 0.0, 1.5, 3.0]
        );
        assert_eq!(
            backend.unary(UnaryKind::Neg, &x),
            vec![2.0, 0.5, 0.0, -1.5, -3.0]
        );
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        assert_eq!(
            backend.binary(BinaryKind::Add, &a, &b),
            vec![11.0, 22.0, 33.0, 44.0]
        );
        let (m, k, n) = (2, 3, 2);
        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 + 1.0).collect();
        assert_eq!(
            backend.matmul(&a, &b, m, k, n),
            naive_matmul(&a, &b, m, k, n)
        );

        // Direct *_dev round trip plus foreign-buffer rejection.
        let xd = backend.alloc_from_host(&x).unwrap();
        assert_eq!(xd.device(), Device::Cuda(0));
        assert_eq!(xd.len(), x.len());
        let rd = backend.unary_dev(UnaryKind::Relu, xd.as_ref()).unwrap();
        assert_eq!(
            backend.copy_to_host(rd.as_ref()).unwrap(),
            vec![0.0, 0.0, 0.0, 1.5, 3.0]
        );
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
        let y = x
            .matmul(&w_true)
            .unwrap()
            .add(&Tensor::from_vec(vec![0.5], &[1]).unwrap())
            .unwrap();

        let run = |device: Option<Device>| -> (Vec<f32>, Vec<f32>, f32, f32) {
            let place = |t: Tensor| match device {
                Some(d) => t.to_device(d).unwrap(),
                None => t,
            };
            let xd = place(x.clone());
            let yd = place(y.clone());
            let lr = place(Tensor::scalar(0.1));
            let mut w = place(Tensor::from_vec(vec![0.0, 0.0], &[2, 1]).unwrap())
                .requires_grad_(true)
                .unwrap();
            let mut b = place(Tensor::from_vec(vec![0.0], &[1]).unwrap())
                .requires_grad_(true)
                .unwrap();
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
                w = w
                    .detach_copy()
                    .sub(&gw.mul(&lr).unwrap())
                    .unwrap()
                    .requires_grad_(true)
                    .unwrap();
                b = b
                    .detach_copy()
                    .sub(&gb.mul(&lr).unwrap())
                    .unwrap()
                    .requires_grad_(true)
                    .unwrap();
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
        assert!(
            last < first * 0.05,
            "loss did not converge on device: {first} -> {last}"
        );
        let (w_cpu, b_cpu, _, _) = run(None);
        for (d, c) in w_dev.iter().zip(w_cpu.iter()) {
            assert!((d - c).abs() < 1e-4, "w: device {d} vs cpu {c}");
        }
        for (d, c) in b_dev.iter().zip(b_cpu.iter()) {
            assert!((d - c).abs() < 1e-4, "b: device {d} vs cpu {c}");
        }
    }

    const TOL: f32 = 1e-6;

    fn close(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what} length");
        for (i, (&d, &c)) in a.iter().zip(b.iter()).enumerate() {
            assert!((d - c).abs() < TOL, "{what}[{i}]: device {d} vs cpu {c}");
        }
    }

    // Forward + backward parity vs CPU for the row kernels and activations,
    // on a [batch, seq, dim] tensor so the last-dim softmax path and the
    // keepdim sum_dim used by norms are both exercised.
    #[test]
    fn gpu_softmax_gelu_activations_match_cpu_forward_and_grad() {
        if !is_available() {
            return;
        }
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(_) => return,
        };
        let dev = Device::Cuda(0);
        register_backend(dev, backend);
        let data: Vec<f32> = (0..120)
            .map(|i| ((i as f32) * 0.37).sin() * 2.0 - 0.1)
            .collect();
        let shape = [4usize, 5, 6];
        let coef: Vec<f32> = (0..120).map(|i| ((i as f32) * 0.11).cos()).collect();
        let ch = coef.clone();
        let wd = Tensor::from_vec(coef, &shape)
            .unwrap()
            .to_device(dev)
            .unwrap();
        let wc = Tensor::from_vec(ch.clone(), &shape).unwrap();

        let run_op = |f: &dyn Fn(&Tensor) -> Tensor| -> (Vec<f32>, Vec<f32>) {
            let xd = Tensor::from_vec(data.clone(), &shape)
                .unwrap()
                .to_device(dev)
                .unwrap()
                .requires_grad_(true)
                .unwrap();
            let out = f(&xd);
            assert_eq!(out.device(), dev, "output left the device");
            out.mul(&wd).unwrap().sum().backward();
            let gx = xd.grad().unwrap();
            assert_eq!(gx.device(), dev, "grad left the device");
            let gv = gx.to_vec();
            let xc = Tensor::from_vec(data.clone(), &shape)
                .unwrap()
                .requires_grad_(true)
                .unwrap();
            let oc = f(&xc);
            oc.mul(&wc).unwrap().sum().backward();
            (oc.to_vec(), gv)
        };

        let (y, gx) = run_op(&|t| t.softmax(2).unwrap());
        let (yc, gxc) = run_op_cpu(&data, &shape, &ch, &|t| t.softmax(2).unwrap());
        close(&y, &yc, "softmax");
        close(&gx, &gxc, "softmax grad");
        // Every softmax row sums to 1.
        for row in y.chunks(6) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < TOL, "row sum {s}");
        }

        let (y, gx) = run_op(&|t| t.log_softmax(2).unwrap());
        let (yc, gxc) = run_op_cpu(&data, &shape, &ch, &|t| t.log_softmax(2).unwrap());
        close(&y, &yc, "log_softmax");
        close(&gx, &gxc, "log_softmax grad");

        let (y, gx) = run_op(&|t| t.gelu());
        let (yc, gxc) = run_op_cpu(&data, &shape, &ch, &|t| t.gelu());
        close(&y, &yc, "gelu");
        close(&gx, &gxc, "gelu grad");

        let (y, gx) = run_op(&|t| t.silu());
        let (yc, gxc) = run_op_cpu(&data, &shape, &ch, &|t| t.silu());
        close(&y, &yc, "silu");
        close(&gx, &gxc, "silu grad");

        let (y, gx) = run_op(&|t| t.sigmoid());
        let (yc, gxc) = run_op_cpu(&data, &shape, &ch, &|t| t.sigmoid());
        close(&y, &yc, "sigmoid");
        close(&gx, &gxc, "sigmoid grad");

        // sum_dim over every dim of the [4,5,6] buffer, keepdim layout.
        let xdev = Tensor::from_vec(data.clone(), &shape)
            .unwrap()
            .to_device(dev)
            .unwrap();
        let xcpu = Tensor::from_vec(data.clone(), &shape).unwrap();
        for dim in 0..3 {
            let got = xdev.sum_dim(dim, true).unwrap();
            assert_eq!(got.device(), dev);
            let mut want_shape = shape.to_vec();
            want_shape[dim] = 1;
            assert_eq!(got.shape(), &want_shape[..]);
            close(
                &got.to_vec(),
                &xcpu.sum_dim(dim, true).unwrap().to_vec(),
                "sum_dim",
            );
        }
    }

    // CPU twin of run_op above so each op is compared against identical host
    // math rather than a duplicated formula.
    fn run_op_cpu(
        data: &[f32],
        shape: &[usize],
        coef: &[f32],
        f: &dyn Fn(&Tensor) -> Tensor,
    ) -> (Vec<f32>, Vec<f32>) {
        let x = Tensor::from_vec(data.to_vec(), shape)
            .unwrap()
            .requires_grad_(true)
            .unwrap();
        let out = f(&x);
        out.mul(&Tensor::from_vec(coef.to_vec(), shape).unwrap())
            .unwrap()
            .sum()
            .backward();
        (out.to_vec(), x.grad().unwrap().to_vec())
    }

    // Mini transformer-style block (linear -> gelu -> linear -> cross entropy
    // via log_softmax) trained fully resident on the GPU: every intermediate,
    // gradient, and parameter stays on Device::Cuda(0) and the loss converges.
    #[test]
    fn gpu_mini_block_forward_backward_fully_resident_converges() {
        if !is_available() {
            return;
        }
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(_) => return,
        };
        let dev = Device::Cuda(0);
        register_backend(dev, backend);

        let (batch, din, dhid, classes) = (8usize, 8, 16, 10);
        let lin = |n: usize, m: usize, scale: f32| -> Vec<f32> {
            (0..n * m)
                .map(|i| (((i % 17) as f32 / 17.0) - 0.5) * scale)
                .collect()
        };
        let labels: Vec<usize> = (0..batch).map(|i| i % classes).collect();
        let mut onehot = vec![0f32; batch * classes];
        for (i, &l) in labels.iter().enumerate() {
            onehot[i * classes + l] = 1.0;
        }

        let run = |device: Option<Device>| -> f32 {
            let pl = |t: Tensor| match device {
                Some(d) => t.to_device(d).unwrap(),
                None => t,
            };
            let xd = pl(Tensor::from_vec(lin(batch, din, 1.0), &[batch, din]).unwrap());
            let targets = pl(Tensor::from_vec(onehot.clone(), &[batch, classes]).unwrap());
            let lr = pl(Tensor::scalar(0.5));
            let mut w1 = pl(Tensor::from_vec(lin(din, dhid, 0.5), &[din, dhid]).unwrap())
                .requires_grad_(true)
                .unwrap();
            let mut b1 = pl(Tensor::from_vec(vec![0f32; dhid], &[dhid]).unwrap())
                .requires_grad_(true)
                .unwrap();
            let mut w2 = pl(Tensor::from_vec(lin(dhid, classes, 0.5), &[dhid, classes]).unwrap())
                .requires_grad_(true)
                .unwrap();
            let mut b2 = pl(Tensor::from_vec(vec![0f32; classes], &[classes]).unwrap())
                .requires_grad_(true)
                .unwrap();
            let (mut first, mut last) = (f32::NAN, f32::NAN);
            for step in 0..150 {
                let hid = xd.matmul(&w1).unwrap().add(&b1).unwrap().gelu();
                let logits = hid.matmul(&w2).unwrap().add(&b2).unwrap();
                if let Some(d) = device {
                    // If log_softmax fell back to the host, CE would silently
                    // realign and this assertion would fire.
                    assert_eq!(logits.device(), d, "logits left the device");
                }
                let loss = ferro_core::nn::cross_entropy(&logits, &targets).unwrap();
                loss.backward();
                let params = [
                    (w1.grad().unwrap(), "w1"),
                    (b1.grad().unwrap(), "b1"),
                    (w2.grad().unwrap(), "w2"),
                    (b2.grad().unwrap(), "b2"),
                ];
                for (g, name) in &params {
                    if let Some(d) = device {
                        assert_eq!(g.device(), d, "{name} grad left the device");
                    }
                }
                let step_t = |p: &Tensor, g: &Tensor| {
                    p.detach_copy()
                        .sub(&g.mul(&lr).unwrap())
                        .unwrap()
                        .requires_grad_(true)
                        .unwrap()
                };
                w1 = step_t(&w1, &w1.grad().unwrap());
                b1 = step_t(&b1, &b1.grad().unwrap());
                w2 = step_t(&w2, &w2.grad().unwrap());
                b2 = step_t(&b2, &b2.grad().unwrap());
                let l = loss.item();
                if step == 0 {
                    first = l;
                }
                last = l;
            }
            if let Some(d) = device {
                for (p, name) in [&w1, &b1, &w2, &b2].map(|p| (p, "")) {
                    let _ = name;
                    assert_eq!(p.device(), d);
                }
            }
            assert!(
                last < first * 0.25,
                "loss did not converge: {first} -> {last}"
            );
            first.min(last)
        };

        run(Some(dev));
        run(None);
    }

    /// G6 structural proof: after warm-up, a repeated forward "layer" step
    /// (matmul -> bias add -> relu) makes ZERO fresh device allocations per
    /// step — every buffer it needs is served from the caching allocator's
    /// freelist. Also checks the recycling allocator is bit-identical to the
    /// pass-through baseline, so correctness is not traded for the win.
    #[test]
    fn g6_zero_fresh_allocations_per_step_after_warmup() {
        if !is_available() {
            return;
        }
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(_) => return,
        };

        // A small "layer": y = relu(x @ w + b_bcast). Fixed shapes every step,
        // exactly as a real training loop re-requests the same buffers.
        let (m, k, n) = (16usize, 32usize, 16usize);
        let x = backend
            .alloc_from_host(&(0..m * k).map(|i| (i as f32 % 7.0) - 3.0).collect::<Vec<_>>())
            .unwrap();
        let w = backend
            .alloc_from_host(&(0..k * n).map(|i| (i as f32 % 5.0) - 2.0).collect::<Vec<_>>())
            .unwrap();
        let b = backend
            .alloc_from_host(&(0..m * n).map(|i| (i as f32 % 3.0) - 1.0).collect::<Vec<_>>())
            .unwrap();

        let step = |backend: &CudaBackend| -> Vec<f32> {
            // Each intermediate buffer is dropped at the end of the closure,
            // returning its slice to the freelist for the next step to reuse.
            let mm = backend.matmul_dev(x.as_ref(), w.as_ref(), m, k, n, false, false).unwrap();
            let bias = backend.binary_dev(BinaryKind::Add, mm.as_ref(), b.as_ref()).unwrap();
            let act = backend.unary_dev(UnaryKind::Relu, bias.as_ref()).unwrap();
            backend.copy_to_host(act.as_ref()).unwrap()
        };

        // Warm up: first two steps populate the freelist (kernel compile +
        // first allocation of each distinct size).
        let warm = step(&backend);
        let _ = step(&backend);

        // Measured window: allocations must be fully served from the pool.
        let before = backend.alloc_stats();
        let mut last = Vec::new();
        for _ in 0..10 {
            last = step(&backend);
        }
        let delta = backend.alloc_stats().since(&before);

        assert_eq!(
            delta.driver_allocs, 0,
            "expected zero fresh driver allocations across 10 steps after warm-up, got {} \
             (requests={}, hits={}); pool did not fully absorb the step's buffers",
            delta.driver_allocs, delta.requests, delta.hits
        );
        assert!(
            delta.requests > 0 && delta.hits == delta.requests,
            "every request in the measured window must be a freelist hit: {delta:?}"
        );
        assert_eq!(warm, last, "results must be stable across steps");

        // Correctness: the recycling allocator must match a pass-through
        // backend bit-for-bit on the same inputs (re-zero-on-hit is correct).
        let plain = match CudaBackend::new_passthrough(0) {
            Ok(b) => Arc::new(b),
            Err(_) => return,
        };
        let px = plain.alloc_from_host(&backend.copy_to_host(x.as_ref()).unwrap()).unwrap();
        let pw = plain.alloc_from_host(&backend.copy_to_host(w.as_ref()).unwrap()).unwrap();
        let pb = plain.alloc_from_host(&backend.copy_to_host(b.as_ref()).unwrap()).unwrap();
        let mm = plain.matmul_dev(px.as_ref(), pw.as_ref(), m, k, n, false, false).unwrap();
        let bias = plain.binary_dev(BinaryKind::Add, mm.as_ref(), pb.as_ref()).unwrap();
        let act = plain.unary_dev(UnaryKind::Relu, bias.as_ref()).unwrap();
        let plain_out = plain.copy_to_host(act.as_ref()).unwrap();
        assert_eq!(
            plain_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            last.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "caching allocator changed numerics vs pass-through baseline"
        );

        // The pass-through backend must NOT recycle: its pool stays empty.
        drop(mm);
        drop(bias);
        drop(act);
        assert_eq!(plain.pooled_buffers(), 0, "pass-through allocator must not pool");
    }

    /// Regression: matmul over an empty reduction dim (k==0) must return
    /// all-zeros, even when the allocator hands back a recycled (dirty) slice.
    /// Guards the alloc_uninit fast path from leaking stale buffer contents on
    /// the GEMM-skip path (bot review finding on PR #12).
    #[test]
    fn matmul_k_zero_is_zeroed_even_with_dirty_pool() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => Arc::new(b),
            Err(_) => return,
        };
        let (m, n) = (4usize, 3usize);
        // Dirty the m*n bin: allocate a buffer full of non-zero values, drop it
        // so its slice returns to the freelist for the next m*n request.
        let dirty = backend.alloc_from_host(&vec![7.0f32; m * n]).unwrap();
        drop(dirty);
        // k==0 GEMM: empty operands, output must be all zeros, not the 7.0s.
        let a = backend.alloc_from_host(&[] as &[f32]).unwrap();
        let b = backend.alloc_from_host(&[] as &[f32]).unwrap();
        let c = backend
            .matmul_dev(a.as_ref(), b.as_ref(), m, 0, n, false, false)
            .unwrap();
        assert_eq!(
            backend.copy_to_host(c.as_ref()).unwrap(),
            vec![0.0f32; m * n]
        );
    }
}
