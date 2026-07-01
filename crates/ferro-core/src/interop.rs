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

use crate::tensor::Tensor;
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
