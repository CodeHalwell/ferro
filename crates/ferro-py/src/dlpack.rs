//! DLPack producer/consumer glue for the Python `Tensor`.
//!
//! Export: `__dlpack__` builds a `DLManagedTensor` and wraps it in a
//! PyCapsule named "dltensor". CPU tensors export an owned heap copy of their
//! row-major f32 data; CUDA-resident tensors are exported ZERO-COPY as
//! kDLCUDA: the DLTensor's data pointer is the backend allocation's base and
//! any view offset is carried in `byte_offset`. OWNERSHIP: an exported CUDA
//! view BORROWS the source Tensor's Arc-shared storage - it holds one strong
//! reference, so GPU memory is never freed while the capsule lives, and the
//! deleter only releases that reference (it never calls cudaFree on memory an
//! alive Tensor still owns). The consumer reads the device buffer directly.
//!
//! Import: `from_dlpack` pulls the capsule from an object's `__dlpack__`,
//! then calls the source's `deleter`. kDLCPU input is copied into host
//! storage. kDLCUDA input is NOT imported zero-copy (the pointer may come
//! from any same-context producer): we download the described span to the
//! host with cuMemcpyDtoH, gather through the source strides, then do an HTOD
//! copy through ferro_cuda's registered backend so the result is a normal
//! device-resident ferro tensor. This assumes the incoming pointer belongs to
//! the primary context of the stated device (true for same-process producers).

use std::ffi::{c_char, c_void, CStr};
use std::os::raw::c_int;

use cudarc::driver::result::memcpy_dtoh_sync;
use ferro_core::{Device, Tensor as CoreTensor};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

// DLPack C ABI structs (subset sufficient for CPU f32 tensors).

const K_DL_CPU: i32 = 1;
const K_DL_CUDA: i32 = 2;
const K_DL_FLOAT: u8 = 2;

#[repr(C)]
struct DLDevice {
    device_type: c_int,
    device_id: c_int,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: c_int,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Backing allocations for an exported DLManagedTensor. Stored in
/// `manager_ctx` and reclaimed by the deleter. `Host` owns a heap copy;
/// `Device` borrows the source tensor's Arc storage (see module docs).
struct ManagerCtx {
    data: Exported,
    shape: Vec<i64>,
}

enum Exported {
    Host(Vec<f32>),
    /// Borrows the source tensor's Arc storage; dropping it never frees GPU
    /// memory an alive Tensor still shares. Never read: kept alive on purpose.
    #[allow(dead_code)]
    Device(ferro_core::interop::DeviceView),
}
unsafe extern "C" fn managed_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    // Reclaim both allocations made at export: the DLManagedTensor box
    // itself and the ManagerCtx box. For a device view this drops the Arc
    // reference on the borrowed storage - GPU memory is freed only when the
    // last Tensor sharing it is gone, never by this deleter directly.
    let managed_box = Box::from_raw(managed);
    let ctx = managed_box.manager_ctx as *mut ManagerCtx;
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

/// PyCapsule destructor: if the capsule still holds an un-consumed
/// "dltensor" (the consumer renames it after taking ownership), free it.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    let name = ffi::PyCapsule_GetName(capsule);
    if name.is_null() {
        return;
    }
    if CStr::from_ptr(name).to_bytes() != b"dltensor" {
        // Renamed to "used_dltensor": ownership transferred to the consumer.
        return;
    }
    let ptr = ffi::PyCapsule_GetPointer(capsule, name) as *mut DLManagedTensor;
    if ptr.is_null() {
        return;
    }
    if let Some(deleter) = (*ptr).deleter {
        deleter(ptr);
    }
}

/// DLTensor header for a kDLCUDA f32 tensor. `data` is the allocation base;
/// a view offset is carried in `byte_offset` (must be f32-aligned, i.e.
/// divisible by 4). Pure so unit tests can pin the ABI fields without a GPU.
fn cuda_dl_tensor(data: *mut c_void, device_id: c_int, byte_offset: u64, dims: Vec<i64>) -> (DLTensor, Vec<i64>) {
    let mut shape = dims;
    let ndim = shape.len() as c_int;
    let t = DLTensor {
        data,
        device: DLDevice { device_type: K_DL_CUDA, device_id },
        ndim,
        dtype: DLDataType { code: K_DL_FLOAT, bits: 32, lanes: 1 },
        shape: shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        byte_offset,
    };
    (t, shape)
}

/// byte_offset for a view starting at element `offset_elems`; overflow-free
/// by construction via u64 checked math (mirrors the % 4 rule on import).
fn view_byte_offset(offset_elems: usize) -> PyResult<u64> {
    u64::try_from(offset_elems)
        .ok()
        .and_then(|o| o.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| PyValueError::new_err("DLPack view offset overflows u64"))
}

/// Build a "dltensor" capsule owning a copy of `data` with the given `shape`.
///
/// We use the raw `PyCapsule_New` so the capsule's pointer IS the
/// `DLManagedTensor*` directly, as the DLPack protocol requires (pyo3's safe
/// `new_with_destructor` would box the value behind its own wrapper).
pub fn export_capsule<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();

    let mut ctx = Box::new(ManagerCtx { data: Exported::Host(data), shape: dims });

    let Exported::Host(host) = &ctx.data else { unreachable!() };
    let data_ptr = host.as_ptr() as *mut c_void;
    let shape_ptr = ctx.shape.as_mut_ptr();
    let ndim = ctx.shape.len() as c_int;

    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: data_ptr,
            device: DLDevice { device_type: K_DL_CPU, device_id: 0 },
            ndim,
            dtype: DLDataType { code: K_DL_FLOAT, bits: 32, lanes: 1 },
            shape: shape_ptr,
            // NULL strides == row-major contiguous, which our copy always is.
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: Some(managed_deleter),
    });
    let managed_ptr = Box::into_raw(managed);

    let ctx_ptr = Box::into_raw(ctx);
    unsafe {
        (*managed_ptr).manager_ctx = ctx_ptr as *mut c_void;
    }

    unsafe {
        let name = c"dltensor";
        let cap = ffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            name.as_ptr(),
            Some(capsule_destructor),
        );
        if cap.is_null() {
            // Reclaim on failure so we don't leak.
            managed_deleter(managed_ptr);
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, cap))
    }
}

/// Build a "dltensor" capsule that ZERO-COPY exports a device-resident
/// contiguous f32 tensor as kDLCUDA. The capsule borrows the tensor's Arc
/// storage; see the module docs for the ownership contract.
pub fn export_device_capsule<'py>(
    py: Python<'py>,
    inner: &CoreTensor,
) -> PyResult<Bound<'py, PyAny>> {
    let view = inner.dlpack_device_view().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (ptr, ordinal) =
        ferro_cuda::exported_view(view.device_buffer()).map_err(PyValueError::new_err)?;
    if ptr == 0 {
        return Err(PyValueError::new_err("CUDA buffer has null device pointer"));
    }
    let byte_offset = view_byte_offset(view.offset_elems())?;
    if byte_offset % 4 != 0 {
        return Err(PyValueError::new_err("view offset is not f32-aligned"));
    }
    let dims: Vec<i64> = inner.shape().iter().map(|&d| d as i64).collect();

    let mut ctx = Box::new(ManagerCtx {
        data: Exported::Device(view),
        shape: Vec::new(),
    });
    let (dl_tensor, shape) =
        cuda_dl_tensor(ptr as *mut c_void, ordinal as c_int, byte_offset, dims);
    ctx.shape = shape;

    let shape_ptr = ctx.shape.as_mut_ptr();
    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor { ..dl_tensor },
        manager_ctx: std::ptr::null_mut(),
        deleter: Some(managed_deleter),
    });
    finish_capsule(py, managed, shape_ptr, ctx)
}

fn finish_capsule<'py>(
    py: Python<'py>,
    mut managed: Box<DLManagedTensor>,
    shape_ptr: *mut i64,
    ctx: Box<ManagerCtx>,
) -> PyResult<Bound<'py, PyAny>> {
    managed.dl_tensor.shape = shape_ptr;
    let managed_ptr = Box::into_raw(managed);
    let ctx_ptr = Box::into_raw(ctx);
    unsafe {
        (*managed_ptr).manager_ctx = ctx_ptr as *mut c_void;
    }
    unsafe {
        let name = c"dltensor";
        let cap = ffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            name.as_ptr(),
            Some(capsule_destructor),
        );
        if cap.is_null() {
            // Reclaim on failure so we don't leak.
            managed_deleter(managed_ptr);
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, cap))
    }
}

/// `__dlpack__` dispatch: zero-copy kDLCUDA for device-resident tensors,
/// owned host copy otherwise.
pub fn export_for<'py>(py: Python<'py>, inner: &CoreTensor) -> PyResult<Bound<'py, PyAny>> {
    if inner.device() != Device::Cpu {
        return export_device_capsule(py, inner);
    }
    let (data, shape) = inner.to_contiguous();
    export_capsule(py, data, shape)
}

/// `__dlpack_device__` value: (kDLCPU, 0) or (kDLCUDA, ordinal).
pub fn dlpack_device_for<'py>(py: Python<'py>, inner: &CoreTensor) -> Bound<'py, PyTuple> {
    match inner.device() {
        Device::Cuda(n) => PyTuple::new(py, [K_DL_CUDA, n as i32]).unwrap(),
        Device::Cpu => PyTuple::new(py, [K_DL_CPU, 0]).unwrap(),
    }
}

/// Consume an object exposing `__dlpack__`, copying into a new ferro tensor.
pub fn import_from_dlpack(obj: &Bound<'_, PyAny>) -> PyResult<CoreTensor> {
    let capsule_obj = obj.call_method0("__dlpack__")?;
    let capsule = capsule_obj.downcast::<PyCapsule>().map_err(|_| {
        PyValueError::new_err("__dlpack__ did not return a PyCapsule")
    })?;

    unsafe {
        let name = ffi::PyCapsule_GetName(capsule.as_ptr());
        if name.is_null() || CStr::from_ptr(name).to_bytes() != b"dltensor" {
            return Err(PyValueError::new_err(
                "expected an unconsumed \"dltensor\" DLPack capsule",
            ));
        }
        let managed = ffi::PyCapsule_GetPointer(capsule.as_ptr(), name) as *mut DLManagedTensor;
        if managed.is_null() {
            return Err(PyValueError::new_err("null dltensor capsule"));
        }

        let tensor = read_managed(managed)?;

        // Rename the capsule so its destructor won't also free the managed
        // tensor, then invoke the producer's deleter now that we've copied.
        let used = c"used_dltensor";
        if ffi::PyCapsule_SetName(capsule.as_ptr(), used.as_ptr() as *const c_char) != 0 {
            // Rename failed: the capsule destructor still owns the managed
            // tensor, so calling the deleter here would double-free. Leave
            // ownership with the capsule and surface the Python error.
            return Err(PyErr::take(obj.py()).unwrap_or_else(|| {
                PyValueError::new_err("failed to rename consumed DLPack capsule")
            }));
        }
        if let Some(deleter) = (*managed).deleter {
            deleter(managed);
        }
        Ok(tensor)
    }
}

unsafe fn read_managed(managed: *mut DLManagedTensor) -> PyResult<CoreTensor> {
    let t = &(*managed).dl_tensor;

    if t.device.device_type != K_DL_CPU && t.device.device_type != K_DL_CUDA {
        return Err(PyValueError::new_err(
            "only CPU (kDLCPU) and CUDA (kDLCUDA) DLPack tensors are supported",
        ));
    }
    if t.dtype.code != K_DL_FLOAT || t.dtype.bits != 32 || t.dtype.lanes != 1 {
        return Err(PyValueError::new_err("only float32 DLPack tensors are supported"));
    }

    if t.ndim < 0 {
        return Err(PyValueError::new_err("DLPack tensor has negative ndim"));
    }
    let ndim = t.ndim as usize;
    if ndim > 0 && t.shape.is_null() {
        return Err(PyValueError::new_err("DLPack tensor has null shape"));
    }
    // Validate dims before any arithmetic: a negative or oversized dim must be
    // rejected rather than wrapped into a huge usize.
    let mut shape: Vec<usize> = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let d = *t.shape.add(i);
        if d < 0 {
            return Err(PyValueError::new_err(format!(
                "DLPack tensor has negative extent ({d}) in dimension {i}"
            )));
        }
        shape.push(d as usize);
    }
    let numel: usize = if ndim == 0 {
        1
    } else {
        shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| {
            PyValueError::new_err("DLPack tensor element count overflows usize")
        })?
    };
    // Validate strides (in elements) by bounding every gatherable offset:
    // the min and max element offset over all index combinations must stay
    // within [0, numel) relative to base, or pointer arithmetic below could
    // read outside the producer's buffer.
    let mut lo: i128 = 0;
    let mut hi: i128 = numel.saturating_sub(1) as i128;
    if !t.strides.is_null() {
        lo = 0;
        hi = 0;
        for i in 0..ndim {
            let s = *t.strides.add(i);
            let span = (shape[i] as i128 - 1) * s as i128;
            lo += span.min(0);
            hi += span.max(0);
        }
        if lo < 0 || hi >= numel as i128 {
            return Err(PyValueError::new_err(
                "DLPack tensor strides reach outside the described element range",
            ));
        }
    }
    // Zero-element tensors may carry a null data pointer; never touch it
    // (pointer arithmetic and from_raw_parts require non-null even for len 0).
    if numel == 0 {
        return CoreTensor::from_contiguous(&[], &shape)
            .map_err(|e| PyValueError::new_err(e.to_string()));
    }
    if t.data.is_null() {
        return Err(PyValueError::new_err("DLPack tensor has null data pointer"));
    }

    if t.byte_offset as usize % std::mem::size_of::<f32>() != 0 {
        return Err(PyValueError::new_err("DLPack byte_offset is not f32-aligned"));
    }

    if t.device.device_type == K_DL_CUDA {
        return import_cuda(t, ndim, &shape, numel, lo, hi);
    }

    let base = (t.data as *const u8).add(t.byte_offset as usize) as *const f32;

    // Gather elements honoring the source strides (in elements) if present;
    // NULL strides means row-major contiguous.
    let data: Vec<f32> = if t.strides.is_null() {
        std::slice::from_raw_parts(base, numel).to_vec()
    } else {
        let strides: Vec<isize> = (0..ndim).map(|i| *t.strides.add(i) as isize).collect();
        let mut out = Vec::with_capacity(numel);
        let mut idx = vec![0usize; ndim];
        for _ in 0..numel {
            let mut off: isize = 0;
            for d in 0..ndim {
                off += idx[d] as isize * strides[d];
            }
            out.push(*base.offset(off));
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    };

    CoreTensor::from_contiguous(&data, &shape)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// kDLCUDA import: NOT zero-copy. Download the described element span from
/// the producer's device pointer with a synchronous cuMemcpyDtoH (requires
/// the pointer to live in the device's primary context - true for
/// same-process producers), gather through the source strides on the host,
/// then HTOD copy through ferro_cuda's registered backend so the result is a
/// normal device-resident ferro tensor.
unsafe fn import_cuda(
    t: &DLTensor,
    ndim: usize,
    shape: &[usize],
    numel: usize,
    lo: i128,
    hi: i128,
) -> PyResult<CoreTensor> {
    if numel == 0 {
        // Empty tensors never touch the pointer; keep them off-device.
        return CoreTensor::from_contiguous(&[], shape)
            .map_err(|e| PyValueError::new_err(e.to_string()));
    }
    let ordinal = t.device.device_id;
    if ordinal < 0 {
        return Err(PyValueError::new_err("DLPack tensor has negative device_id"));
    }
    ensure_backend(ordinal)?;
    let ctx = cudarc::driver::CudaContext::new(ordinal as usize)
        .map_err(|e| PyValueError::new_err(format!("failed to open CUDA context: {e}")))?;
    ctx.bind_to_thread()
        .map_err(|e| PyValueError::new_err(format!("failed to bind CUDA context: {e}")))?;

    let first = lo + t.byte_offset as i128 / 4;
    let count = (hi - lo) as usize + 1;
    let src = (t.data as usize as u64).wrapping_add((first * 4) as u64);
    let mut scratch = vec![0f32; count];
    memcpy_dtoh_sync(&mut scratch, src)
        .map_err(|e| PyValueError::new_err(format!("cuMemcpyDtoH failed: {e}")))?;

    // Gather through the source strides (in elements); NULL strides means
    // row-major contiguous, which is a straight copy of the scratch window.
    let data: Vec<f32> = if t.strides.is_null() {
        scratch
    } else {
        let strides: Vec<isize> = (0..ndim).map(|i| *t.strides.add(i) as isize).collect();
        let base_elem = lo; // relative to the download start
        let mut out = Vec::with_capacity(numel);
        let mut idx = vec![0usize; ndim];
        for _ in 0..numel {
            let mut off: i128 = 0;
            for d in 0..ndim {
                off += idx[d] as i128 * strides[d] as i128;
            }
            out.push(scratch[(off as i128 - base_elem) as usize]);
            for d in (0..ndim).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    };

    // HTOD copy through the backend: the result shares nothing with the
    // source buffer and the producer's deleter can run immediately after.
    CoreTensor::from_contiguous(&data, shape)
        .and_then(|host| host.to_device(Device::Cuda(ordinal as u32)))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn ensure_backend(ordinal: c_int) -> PyResult<()> {
    use ferro_core::dispatch::backend_for;
    if backend_for(Device::Cuda(ordinal as u32)).is_ok() {
        return Ok(());
    }
    ferro_cuda::install(ordinal as u32).map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    // read_managed only reads dl_tensor before the deleter runs, so tests can
    // pass a zeroed DLManagedTensor plus raw shape/strides scratch buffers.
    // PyErr construction needs a live interpreter, hence with_gil everywhere.
    struct Fixture {
        managed: Box<DLManagedTensor>,
        shape: Vec<i64>,
        strides: Vec<i64>,
        explicit_strides: bool,
        data: Vec<f32>,
    }

    fn fixture(shape: Vec<i64>, strides: Option<Vec<i64>>, data: Vec<f32>) -> Fixture {
        let mut s = vec![1i64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1].wrapping_mul(shape[i + 1]);
        }
        let explicit_strides = strides.is_some();
        let strides = strides.unwrap_or(s);
        Fixture {
            managed: Box::new(DLManagedTensor {
                dl_tensor: DLTensor {
                    data: std::ptr::null_mut(),
                    device: DLDevice { device_type: K_DL_CPU, device_id: 0 },
                    ndim: shape.len() as c_int,
                    dtype: DLDataType { code: K_DL_FLOAT, bits: 32, lanes: 1 },
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: None,
            }),
            shape,
            strides,
            explicit_strides,
            data,
        }
    }

    fn run(f: &mut Fixture) -> PyResult<CoreTensor> {
        let t = &mut f.managed.dl_tensor;
        t.data = if f.data.is_empty() { std::ptr::null_mut() } else { f.data.as_mut_ptr() as *mut c_void };
        t.shape = f.shape.as_mut_ptr();
        t.strides = if f.explicit_strides { f.strides.as_mut_ptr() } else { std::ptr::null_mut() };
        unsafe { read_managed(f.managed.as_mut() as *mut DLManagedTensor) }
    }

    // The statically-linked interpreter cannot locate its own installation,
    // so seed PYTHONHOME from the system python before initializing it.
    fn init_python() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            // The dynamic cudarc loader needs the driver DLLs on PATH.
            let rt = std::env::var_os("LOCALAPPDATA").map(|p| {
                std::path::PathBuf::from(p).join("Temp").join("cuda-rt")
            });
            if let Some(rt) = rt {
                let mut paths = vec![
                    rt.join("nvidia").join("cuda_nvrtc").join("bin"),
                    rt.join("nvidia").join("cublas").join("bin"),
                ];
                if let Some(p) = std::env::var_os("PATH") {
                    paths.insert(0, std::path::PathBuf::from(p));
                }
                let joined = std::env::join_paths(paths.iter()).unwrap();
                unsafe { std::env::set_var("PATH", joined) };
            }
            if std::env::var_os("PYTHONHOME").is_none() {
                // The abi3-py311 extension links a python3x.dll whose
                // stdlib must match its own major.minor, so accept a
                // candidate prefix only after checking it really contains
                // an encodings package. The WindowsApps "python" stub
                // prints nothing and exits non-zero, hence the probes.
                let mut cands: Vec<String> = Vec::new();
                let probes: [&[&str]; 3] = [
                    &["python", "-c", "import sys; print(sys.base_prefix)"],
                    &["python3", "-c", "import sys; print(sys.base_prefix)"],
                    &["py", "-3.11", "-c", "import sys; print(sys.base_prefix)"],
                ];
                for cmd in probes {
                    let Some((exe, rest)) = cmd.split_first() else { continue };
                    let Ok(out) = std::process::Command::new(exe).args(rest).output() else {
                        continue;
                    };
                    if out.status.success() {
                        let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
                        if !p.is_empty() {
                            cands.push(p);
                        }
                    }
                }
                // Known 3.11 interpreter locations on this machine layout.
                if let Some(uv) = std::env::var_os("APPDATA") {
                    let mut p = std::path::PathBuf::from(uv);
                    p.push("uv");
                    p.push("python");
                    if let Ok(rd) = std::fs::read_dir(&p) {
                        for e in rd.flatten() {
                            let n = e.file_name().to_string_lossy().to_string();
                            if n.starts_with("cpython-3.11") {
                                cands.push(e.path().to_string_lossy().to_string());
                            }
                        }
                    }
                }
                let has_stdlib =
                    |p: &String| std::path::Path::new(p).join("Lib").join("encodings").exists();
                if let Some(p) = cands.iter().find(|p| has_stdlib(p)) {
                    unsafe { std::env::set_var("PYTHONHOME", &**p) };
                }
            }
            pyo3::prepare_freethreaded_python();
        });
    }

    fn err(f: &mut Fixture) -> String {
        init_python();
        Python::with_gil(|_| run(f).err().expect("expected error").to_string())
    }

    fn ok(f: &mut Fixture) -> Vec<f32> {
        init_python();
        Python::with_gil(|_| run(f).expect("expected ok").to_vec())
    }

    #[test]
    fn contiguous_round_trip() {
        let mut f = fixture(vec![2, 3], None, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(ok(&mut f), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_negative_ndim_and_dims() {
        let mut f = fixture(vec![2], None, vec![0.0]);
        f.managed.dl_tensor.ndim = -1;
        assert!(err(&mut f).contains("negative ndim"));

        let mut f = fixture(vec![2, -3], None, vec![0.0; 6]);
        assert!(err(&mut f).contains("negative extent"));
    }

    #[test]
    fn rejects_numel_overflow() {
        let big = 1i64 << 62;
        let mut f = fixture(vec![big, big, big, big], Some(vec![0; 4]), vec![]);
        assert!(err(&mut f).contains("overflows"));
    }

    #[test]
    fn rejects_strides_reaching_outside_buffer() {
        // Each stride alone is within numel, but combined offsets are not.
        let mut f = fixture(vec![2, 2], Some(vec![4, 1]), vec![0.0; 4]);
        assert!(err(&mut f).contains("outside"));
    }

    #[test]
    fn gathers_with_strides() {
        // Row-major (2,3) viewed transposed via strides.
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut f = fixture(vec![3, 2], Some(vec![1, 3]), data);
        assert_eq!(ok(&mut f), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        let mut f = fixture(vec![2, 3], None, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(ok(&mut f), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_bad_device_dtype_and_null_data() {
        // Device type 4 is kDLOneAPI - unsupported here.
        let mut f = fixture(vec![2], None, vec![0.0; 2]);
        f.managed.dl_tensor.device.device_type = 4;
        assert!(err(&mut f).contains("kDLCUDA"));

        let mut f = fixture(vec![2], None, vec![0.0; 2]);
        f.managed.dl_tensor.dtype.bits = 64;
        assert!(err(&mut f).contains("float32"));

        let mut f = fixture(vec![2], None, vec![]);
        f.managed.dl_tensor.data = std::ptr::null_mut();
        assert!(err(&mut f).contains("null data pointer"));
    }

    // The CUDA DLTensor ABI fields are pure data, so a fake pointer pins the
    // producer contract without a GPU: kDLCUDA type, ordinal passthrough,
    // byte_offset carrying the view offset, NULL strides (contiguous).
    #[test]
    fn cuda_dl_tensor_fields_on_fake_buffer() {
        let (t, shape) = cuda_dl_tensor(0xDEAD0000usize as *mut c_void, 1, 12, vec![2, 3]);
        assert_eq!(t.device.device_type, K_DL_CUDA);
        assert_eq!(t.device.device_id, 1);
        assert_eq!(t.byte_offset, 12);
        assert_eq!(t.ndim, 2);
        assert_eq!(t.dtype.code, K_DL_FLOAT);
        assert_eq!(t.dtype.bits, 32);
        assert!(t.strides.is_null());
        assert_eq!(&shape, &[2, 3]);

        let (t, _) = cuda_dl_tensor(std::ptr::null_mut(), 0, 0, vec![]);
        assert_eq!(t.device.device_type, K_DL_CUDA);
        assert_eq!(t.ndim, 0);
    }

    #[test]
    fn view_byte_offset_is_f32_scaled_and_overflow_free() {
        assert_eq!(view_byte_offset(0).unwrap(), 0);
        assert_eq!(view_byte_offset(3).unwrap(), 12);
        assert_eq!(
            view_byte_offset(usize::MAX / 4).unwrap(),
            (usize::MAX / 4) as u64 * 4
        );
        assert!(view_byte_offset(usize::MAX).is_err());
        // The % 4 alignment rule always holds for element-scaled offsets.
        for elems in [0usize, 1, 7, 1023] {
            assert_eq!(view_byte_offset(elems).unwrap() % 4, 0);
        }
    }

    fn gpu_ok() -> bool {
        init_python();
        ferro_cuda::install(0).is_ok()
    }

    // Structural GPU test: export a device-resident tensor zero-copy, pin
    // the emitted header fields, drop the capsule while the tensor lives
    // (deleter must NOT free borrowed GPU memory), then import through the
    // htod path and compare values.
    #[test]
    fn gpu_export_borrows_and_import_htod() {
        if !gpu_ok() {
            return;
        }
        let dev = Device::Cuda(0);
        let t = Python::with_gil(|_| {
            CoreTensor::from_vec((0..6).map(|i| i as f32).collect(), &[6])
                .and_then(|t| t.to_device(dev))
                .map_err(|e| e.to_string())
                .unwrap()
        });

        Python::with_gil(|py| {
            let cap = export_for(py, &t).unwrap();
            let ptr = unsafe {
                ffi::PyCapsule_GetPointer(cap.as_ptr(), c"dltensor".as_ptr())
                    as *mut DLManagedTensor
            };
            assert!(!ptr.is_null());
            let hdr = unsafe { &(*ptr).dl_tensor };
            assert_eq!(hdr.device.device_type, K_DL_CUDA);
            assert_eq!(hdr.device.device_id, 0);
            assert_eq!(hdr.byte_offset, 0);
            assert!(!hdr.data.is_null());
            assert_eq!(unsafe { *hdr.shape }, 6);

            // Import through the htod copy path before the capsule dies.
            let imported = unsafe { read_managed(ptr) }.unwrap();
            assert_eq!(imported.device(), dev);
            assert_eq!(imported.to_vec(), (0..6).map(|i| i as f32).collect::<Vec<_>>());

            // Dropping the unconsumed capsule runs the deleter, which must
            // release only the borrow: the source tensor stays fully usable.
            drop(cap);
            assert_eq!(t.to_vec(), (0..6).map(|i| i as f32).collect::<Vec<_>>());
        });
    }
}
