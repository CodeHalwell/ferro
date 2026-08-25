//! Interop surface for zero-copy/bridged exchange with numpy/torch. The
//! Python-facing DLPack capsule logic lives in the ferro-py crate; here we
//! expose the minimal buffer/metadata access it needs on `Tensor`.
//!
//! The current bridge copies at the boundary: `to_contiguous` materializes a
//! row-major `Vec<f32>` (via `to_vec`) that the producer owns and hands to
//! DLPack, and `from_contiguous` rebuilds a `Tensor` from a borrowed slice.
//! This is always correct regardless of the source strides/offset; a true
//! zero-copy export can be layered on later once a stable pointer accessor is
//! needed.

use std::sync::Arc;

use crate::dispatch::DeviceBuffer;
use crate::tensor::{Storage, StorageCell, Tensor};
use crate::Result;

impl Tensor {
    /// Row-major contiguous data plus its shape, suitable for handing to a
    /// DLPack producer. The returned `Vec` is a fresh owned copy.
    pub fn to_contiguous(&self) -> (Vec<f32>, Vec<usize>) {
        (self.to_vec(), self.shape().to_vec())
    }

    /// Build a tensor from a borrowed row-major f32 slice and shape, copying
    /// the data. Mirrors the DLPack consumer path.
    pub fn from_contiguous(data: &[f32], shape: &[usize]) -> Result<Tensor> {
        Tensor::from_vec(data.to_vec(), shape)
    }
}

/// A borrowed view of a device-resident tensor's storage for zero-copy
/// DLPack export. The held Arc keeps the backend buffer alive while the
/// exported capsule exists; dropping the view never frees GPU memory that an
/// alive Tensor still shares - the buffer dies only with the last reference.
pub struct DeviceView {
    _keep: Arc<StorageCell>,
    buf: *const dyn DeviceBuffer,
    offset_elems: usize,
}

impl DeviceView {
    /// The backend-owned buffer this view borrows. Downcast via `as_any` in
    /// the backend crate to recover the concrete allocation and its pointer.
    pub fn device_buffer(&self) -> &dyn DeviceBuffer {
        unsafe { &*self.buf }
    }

    /// Element offset of the view into the underlying storage; the DLPack
    /// producer turns this into `byte_offset = offset_elems * sizeof(f32)`.
    pub fn offset_elems(&self) -> usize {
        self.offset_elems
    }
}

impl Tensor {
    /// Zero-copy DLPack export source for device-resident contiguous f32
    /// tensors. Non-contiguous or host-stored tensors return Err; those keep
    /// using the copy-based host path (`to_contiguous`).
    pub fn dlpack_device_view(&self) -> Result<DeviceView> {
        if !self.is_contiguous() {
            return Err(crate::Error::Unsupported {
                op: "dlpack_export",
                msg: "non-contiguous views cannot be exported zero-copy".into(),
            });
        }
        let cell = self.0.storage.clone();
        let buf: *const dyn DeviceBuffer = match &cell.data {
            Storage::Device(b) => &**b,
            _ => {
                return Err(crate::Error::Unsupported {
                    op: "dlpack_export",
                    msg: "tensor is not device-resident f32".into(),
                })
            }
        };
        Ok(DeviceView {
            _keep: cell,
            buf,
            offset_elems: self.0.offset,
        })
    }
}
