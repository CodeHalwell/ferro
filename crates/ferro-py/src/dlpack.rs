//! DLPack producer/consumer glue for the Python `Tensor`.
//!
//! Export: `__dlpack__` builds a `DLManagedTensor` that owns a heap copy of the
//! tensor's row-major f32 data and wraps it in a PyCapsule named "dltensor".
//! The consumer (numpy/torch) reads it zero-copy from that buffer and later
//! invokes our `deleter`, which frees the DLManagedTensor and its data.
//!
//! Import: `from_dlpack` pulls the capsule from an object's `__dlpack__`,
//! copies the described data into a fresh ferro tensor, then calls the source's
//! `deleter` so the producer can release its buffer.
//!
//! We always copy at the boundary, so strides/offset of the source never
//! matter for correctness.

use std::ffi::{c_char, c_void, CStr};
use std::os::raw::c_int;

use ferro_core::Tensor as CoreTensor;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

// DLPack C ABI structs (subset sufficient for CPU f32 tensors).

const K_DL_CPU: i32 = 1;
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

/// Owns the heap allocations backing an exported DLManagedTensor. Stored in
/// `manager_ctx` and reclaimed by the deleter.
struct ManagerCtx {
    data: Vec<f32>,
    shape: Vec<i64>,
}

unsafe extern "C" fn managed_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    // Reclaim both allocations made in export_capsule: the DLManagedTensor
    // box itself and the ManagerCtx box holding data/shape.
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

    let mut ctx = Box::new(ManagerCtx { data, shape: dims });

    let data_ptr = ctx.data.as_ptr() as *mut c_void;
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

/// `__dlpack_device__` value for a CPU tensor: (kDLCPU, 0).
pub fn dlpack_device(py: Python<'_>) -> Bound<'_, PyTuple> {
    PyTuple::new(py, [K_DL_CPU, 0]).unwrap()
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

    if t.device.device_type != K_DL_CPU {
        return Err(PyValueError::new_err("only CPU (kDLCPU) DLPack tensors are supported"));
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
    if !t.strides.is_null() {
        let mut lo: i128 = 0;
        let mut hi: i128 = 0;
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
            if std::env::var_os("PYTHONHOME").is_none() {
                if let Ok(out) = std::process::Command::new("python")
                    .args(["-c", "import sys; print(sys.base_prefix)"])
                    .output()
                {
                    let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    if !p.is_empty() {
                        unsafe { std::env::set_var("PYTHONHOME", p) };
                    }
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
        let mut f = fixture(vec![2], None, vec![0.0; 2]);
        f.managed.dl_tensor.device.device_type = 2;
        assert!(err(&mut f).contains("kDLCPU"));

        let mut f = fixture(vec![2], None, vec![0.0; 2]);
        f.managed.dl_tensor.dtype.bits = 64;
        assert!(err(&mut f).contains("float32"));

        let mut f = fixture(vec![2], None, vec![]);
        f.managed.dl_tensor.data = std::ptr::null_mut();
        assert!(err(&mut f).contains("null data pointer"));
    }
}
