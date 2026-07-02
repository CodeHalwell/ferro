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
        ffi::PyCapsule_SetName(capsule.as_ptr(), used.as_ptr() as *const c_char);
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

    let ndim = t.ndim as usize;
    let shape: Vec<usize> = (0..ndim).map(|i| *t.shape.add(i) as usize).collect();
    let numel: usize = shape.iter().product::<usize>().max(if ndim == 0 { 1 } else { 0 });

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
