//! `index_select`: pick rows along `dim` by an explicit usize index list.
//! Forward copies contiguous `inner` blocks; backward scatter-adds grad blocks
//! back to the input shape (add, not overwrite, so duplicate indices accumulate).
//! Device-resident f32 weights with device-resident I64 indices take the
//! backend's `gather_rows_dev` kernel (outer == 1, the embedding shape);
//! everything else runs the host path.

use crate::device::Device;
use crate::dispatch::backend_for;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{device_leaf, Storage, Tensor};

impl Tensor {
    /// Tensor-index variant: `indices` must be a 1-D I64 tensor with in-range,
    /// non-negative entries. Delegates to the slice version (or the device
    /// gather kernel), so the gradient to `self` comes from the same recorded
    /// backward.
    pub fn index_select_t(&self, dim: usize, indices: &Tensor) -> Result<Tensor> {
        if self.device() != indices.device() {
            return Err(Error::DeviceMismatch {
                op: "index_select",
                lhs: self.device(),
                rhs: indices.device(),
            });
        }
        if indices.dtype() != DType::I64 {
            return Err(Error::DtypeMismatch {
                op: "index_select",
                expected: DType::I64,
                got: indices.dtype(),
            });
        }
        if indices.ndim() != 1 {
            return Err(Error::InvalidShape {
                op: "index_select",
                msg: format!("indices must be 1-D, got shape {:?}", indices.shape()),
            });
        }
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "index_select",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let dim_size = self.shape()[dim];
        let inner: usize = self.shape()[dim + 1..].iter().product();
        // Bounds are validated once on the host (a single idx download) for
        // clear errors; the device kernel then trusts them.
        let idx_i64 = indices.to_vec_i64();
        let idx: Vec<usize> = idx_i64
            .iter()
            .copied()
            .map(|i| {
                usize::try_from(i).map_err(|_| Error::InvalidShape {
                    op: "index_select",
                    msg: format!("negative index {i} is not supported"),
                })
            })
            .collect::<Result<_>>()?;
        if let Some(&bad) = idx.iter().find(|&&i| i >= dim_size) {
            return Err(Error::InvalidShape {
                op: "index_select",
                msg: format!("index {bad} out of range for dim {dim} with size {dim_size}"),
            });
        }

        // Resident fast path: whole contiguous f32 weight + i64 index buffer
        // on the same non-CPU device whose backend implements the gather.
        // Requires outer == 1 (the embedding/index-select-dim0 shape).
        let outer: usize = self.shape()[..dim].iter().product();
        if self.device() != Device::Cpu && self.device_resident_whole() {
            if let (Storage::Device(wbuf), Storage::DeviceI64(ibuf)) =
                (&self.0.storage.data, &indices.0.storage.data)
            {
                if outer == 1 {
                    let backend = backend_for(self.0.device)?;
                    match backend.gather_rows_dev(wbuf.as_ref(), ibuf.as_ref(), dim_size, inner) {
                        Ok(out) => {
                            let mut out_shape = self.shape().to_vec();
                            out_shape[dim] = idx.len();
                            let out = device_leaf(out, &out_shape, self.0.device);
                            let in_shape = self.shape().to_vec();
                            let idx = idx.clone();
                            return Ok(out.record_fn(vec![self.clone()], move |g| {
                                // Host scatter-add; accumulate_grad uploads
                                // the cpu gradient to the tensor's device.
                                let gd = g.to_vec();
                                let mut gx = vec![0.0f32; in_shape.iter().product()];
                                let mut src = 0;
                                for &ix in &idx {
                                    let dst = ix * inner;
                                    for j in 0..inner {
                                        gx[dst + j] += gd[src + j];
                                    }
                                    src += inner;
                                }
                                vec![Tensor::from_vec(gx, &in_shape).unwrap()]
                            }));
                        }
                        // A backend without a gather kernel falls through to
                        // the host path below rather than failing the op.
                        Err(crate::error::Error::Unsupported {
                            op: "gather_rows_dev",
                            ..
                        }) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        self.index_select(dim, &idx)
    }

    pub fn index_select(&self, dim: usize, indices: &[usize]) -> Result<Tensor> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(Error::InvalidShape {
                op: "index_select",
                msg: format!("dim {dim} out of range for rank {ndim}"),
            });
        }
        let in_shape = self.shape().to_vec();
        let dim_size = in_shape[dim];
        if let Some(&bad) = indices.iter().find(|&&i| i >= dim_size) {
            return Err(Error::InvalidShape {
                op: "index_select",
                msg: format!("index {bad} out of range for dim {dim} with size {dim_size}"),
            });
        }

        let inner: usize = in_shape[dim + 1..].iter().product();
        let outer: usize = in_shape[..dim].iter().product();
        let data = self.to_vec();
        let mut out_data = Vec::with_capacity(outer * indices.len() * inner);
        for o in 0..outer {
            for &idx in indices {
                let start = (o * dim_size + idx) * inner;
                out_data.extend_from_slice(&data[start..start + inner]);
            }
        }
        let mut out_shape = in_shape.clone();
        out_shape[dim] = indices.len();
        let out = Tensor::from_vec(out_data, &out_shape)?;

        let indices = indices.to_vec();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let gd = g.to_vec();
            let mut gx = vec![0.0f32; in_shape.iter().product()];
            let mut src = 0;
            for o in 0..outer {
                for &idx in &indices {
                    let dst = (o * dim_size + idx) * inner;
                    for j in 0..inner {
                        gx[dst + j] += gd[src + j];
                    }
                    src += inner;
                }
            }
            vec![Tensor::from_vec(gx, &in_shape).unwrap()]
        }))
    }
}
