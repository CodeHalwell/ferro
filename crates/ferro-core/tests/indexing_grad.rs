//! Indexing autograd completion checks: O(1)-magnitude finite differences,
//! exact duplicate-index accumulation (grad = sum of contributions),
//! non-differentiable integer indices, and a counting-backend proof that an
//! index_select backward keeps a device-resident weight on its device.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ferro_core::dispatch::{
    register_backend, Backend, BinaryKind, DeviceBuffer, ReduceKind, UnaryKind,
};
use ferro_core::testkit::{grad_check, grad_check_strict};
use ferro_core::{DType, Device, Error, Tensor};

fn weighted(y: Tensor) -> Tensor {
    let n = y.numel();
    let c = Tensor::from_vec(
        (0..n).map(|i| 0.14 + 0.31 * i as f32).collect::<Vec<_>>(),
        y.shape(),
    )
    .unwrap();
    y.mul(&c).unwrap().sum()
}

#[test]
fn index_select_grad_o1_inputs_with_duplicates() {
    let x = Tensor::from_vec(
        vec![
            0.7, -1.2, 0.4, 1.3, -0.6, 0.2, 0.9, -0.8, 0.5, -0.3, 1.1, 0.6,
        ],
        &[4, 3],
    )
    .unwrap();
    // Duplicate row 2 twice: linear op, so every point is differentiable.
    grad_check_strict(&[x.clone()], |t| {
        weighted(t[0].index_select(0, &[2, 0, 2, 1]).unwrap())
    });
    grad_check(&[x], |t| {
        weighted(t[0].index_select(1, &[1, 1, 0]).unwrap())
    });
}

#[test]
fn index_select_t_grad_o1_inputs() {
    let x = Tensor::from_vec(
        vec![
            0.7, -1.2, 0.4, 1.3, -0.6, 0.2, 0.9, -0.8, 0.5, -0.3, 1.1, 0.6,
        ],
        &[4, 3],
    )
    .unwrap();
    grad_check(&[x.clone()], |t| {
        let ids = Tensor::from_vec_i64(vec![2, 0, 2], &[3]).unwrap();
        weighted(t[0].index_select_t(0, &ids).unwrap())
    });
}

#[test]
fn gather_grad_o1_inputs_with_duplicates() {
    let x = Tensor::from_vec(
        vec![
            0.7, -1.1, 0.4, 1.3, -0.6, 0.2, 0.9, -0.8, 0.5, -0.3, 1.1, 0.6,
        ],
        &[2, 3, 2],
    )
    .unwrap();
    let idx = Tensor::from_vec_i64(vec![1, 1, 0, 2, 0, 2], &[2, 3, 1]).unwrap();
    grad_check(&[x], |t| weighted(t[0].gather(1, &idx).unwrap()));
}

#[test]
fn duplicate_index_rows_accumulate_exactly() {
    // out rows are w[2], w[0], w[2]; row 2 must receive C[0] + C[2].
    let w = Tensor::from_vec(vec![0.3, -1.2, 0.7, 0.4, -0.5, 0.9], &[3, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let c = Tensor::from_vec(vec![0.4, 0.6, -0.2, 0.5, 0.9, -0.7], &[3, 2]).unwrap();
    w.index_select(0, &[2, 0, 2])
        .unwrap()
        .mul(&c)
        .unwrap()
        .sum()
        .backward();
    let got = w.grad().unwrap().to_vec();
    // w row 0 <- C[1]; w row 1 unwritten; w row 2 <- C[0] + C[2].
    let want = [-0.2f32, 0.5, 0.0, 0.0, 1.3, -0.1];
    for (g, e) in got.iter().zip(want) {
        assert!((g - e).abs() < 1e-6, "grad {g} vs expected {e}");
    }

    // Same accumulation through the tensor-index variant / embedding path.
    let w2 = Tensor::from_vec(vec![0.3, -1.2, 0.7, 0.4, -0.5, 0.9], &[3, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let ids = Tensor::from_vec_i64(vec![2, 0, 2], &[3]).unwrap();
    ferro_core::ops_ext::embedding(&w2, &ids)
        .unwrap()
        .mul(&c)
        .unwrap()
        .sum()
        .backward();
    for (g, e) in w2.grad().unwrap().to_vec().iter().zip(want) {
        assert!((g - e).abs() < 1e-6, "embedding grad {g} vs expected {e}");
    }
}

#[test]
fn gather_duplicate_columns_accumulate_exactly() {
    let t = Tensor::from_vec(vec![0.5, -0.3, 1.2, -0.7], &[1, 4])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let idx = Tensor::from_vec_i64(vec![1, 1, 3], &[1, 3]).unwrap();
    let c = Tensor::from_vec(vec![0.8, -0.4, 0.6], &[1, 3]).unwrap();
    t.gather(1, &idx).unwrap().mul(&c).unwrap().sum().backward();
    // Position 1 was gathered twice: grad = 0.8 + (-0.4) = 0.4; position 3 = 0.6.
    let got = t.grad().unwrap().to_vec();
    let want = [0.0f32, 0.4, 0.0, 0.6];
    for (g, e) in got.iter().zip(want) {
        assert!((g - e).abs() < 1e-6, "grad {g} vs expected {e}");
    }
}

#[test]
fn index_tensors_stay_non_differentiable() {
    let ids = Tensor::from_vec_i64(vec![1, 0], &[2]).unwrap();
    assert!(matches!(
        ids.requires_grad_(true),
        Err(Error::DtypeMismatch {
            expected: DType::F32,
            got: DType::I64,
            op: _,
        })
    ));

    // Backward populates grads only on the float weight; indices get none.
    let w = Tensor::from_vec(vec![0.3, -1.2, 0.7, 0.4], &[2, 2])
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let idx = Tensor::from_vec_i64(vec![1, 1, 0], &[3]).unwrap();
    let c = Tensor::from_vec(vec![0.4, 0.6, -0.2, 0.5, 0.9, -0.7], &[3, 2]).unwrap();
    w.index_select_t(0, &idx)
        .unwrap()
        .mul(&c)
        .unwrap()
        .sum()
        .backward();
    assert!(w.grad().is_some());
    assert!(idx.grad().is_none());
}

// --- structural: device-resident weight backward stays on-device ----------
// Own registry slot (Cuda(8)) and counters, separate from tests/device.rs
// (Cuda(9)); process-global state serializes on this poison-tolerant mutex.

const DEV: Device = Device::Cuda(8);

static F32_ALLOCS: AtomicUsize = AtomicUsize::new(0);
static I64_ALLOCS: AtomicUsize = AtomicUsize::new(0);
static TO_HOST_F32_ELEMS: AtomicUsize = AtomicUsize::new(0);
static TO_HOST_I64_ELEMS: AtomicUsize = AtomicUsize::new(0);
static GATHER_ROWS: AtomicUsize = AtomicUsize::new(0);

struct F32Buf(Vec<f32>);

impl DeviceBuffer for F32Buf {
    fn device(&self) -> Device {
        DEV
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct I64Buf(Vec<i64>);

impl DeviceBuffer for I64Buf {
    fn device(&self) -> Device {
        DEV
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn f32_data(buf: &dyn DeviceBuffer) -> &[f32] {
    &buf.as_any()
        .downcast_ref::<F32Buf>()
        .expect("buffer from another backend")
        .0
}

fn i64_data(buf: &dyn DeviceBuffer) -> &[i64] {
    &buf.as_any()
        .downcast_ref::<I64Buf>()
        .expect("buffer from another backend")
        .0
}

struct CountingDevice;

impl Backend for CountingDevice {
    fn unary(&self, _kind: UnaryKind, _x: &[f32]) -> Vec<f32> {
        panic!("host-slice path must not run for device-resident tensors");
    }
    fn binary(&self, _kind: BinaryKind, _a: &[f32], _b: &[f32]) -> Vec<f32> {
        panic!("host-slice path must not run for device-resident tensors");
    }
    fn matmul(&self, _a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        panic!("host-slice path must not run for device-resident tensors");
    }

    fn alloc_from_host(&self, data: &[f32]) -> Result<Box<dyn DeviceBuffer>, ferro_core::Error> {
        F32_ALLOCS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(F32Buf(data.to_vec())))
    }
    fn copy_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<f32>, ferro_core::Error> {
        TO_HOST_F32_ELEMS.fetch_add(buf.len(), Ordering::SeqCst);
        Ok(f32_data(buf).to_vec())
    }
    fn alloc_i64_from_host(
        &self,
        data: &[i64],
    ) -> Result<Box<dyn DeviceBuffer>, ferro_core::Error> {
        I64_ALLOCS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(I64Buf(data.to_vec())))
    }
    fn copy_i64_to_host(&self, buf: &dyn DeviceBuffer) -> Result<Vec<i64>, ferro_core::Error> {
        TO_HOST_I64_ELEMS.fetch_add(buf.len(), Ordering::SeqCst);
        Ok(i64_data(buf).to_vec())
    }
    fn binary_dev(
        &self,
        kind: BinaryKind,
        a: &dyn DeviceBuffer,
        b: &dyn DeviceBuffer,
    ) -> Result<Box<dyn DeviceBuffer>, ferro_core::Error> {
        let f = |x: f32, y: f32| match kind {
            BinaryKind::Add => x + y,
            BinaryKind::Sub => x - y,
            BinaryKind::Mul => x * y,
            BinaryKind::Div => x / y,
        };
        let out = f32_data(a)
            .iter()
            .zip(f32_data(b))
            .map(|(&x, &y)| f(x, y))
            .collect();
        Ok(Box::new(F32Buf(out)))
    }
    fn reduce_dev(
        &self,
        kind: ReduceKind,
        x: &dyn DeviceBuffer,
    ) -> Result<Box<dyn DeviceBuffer>, ferro_core::Error> {
        let v = f32_data(x);
        let s = v.iter().sum::<f32>();
        let out = match kind {
            ReduceKind::Sum => s,
            ReduceKind::Mean => s / v.len() as f32,
        };
        Ok(Box::new(F32Buf(vec![out])))
    }
    fn fill_dev(&self, value: f32, len: usize) -> Result<Box<dyn DeviceBuffer>, ferro_core::Error> {
        Ok(Box::new(F32Buf(vec![value; len])))
    }
    fn binary_bc_dev(
        &self,
        kind: BinaryKind,
        a: &dyn DeviceBuffer,
        sa: &[usize],
        b: &dyn DeviceBuffer,
        sb: &[usize],
        out_shape: &[usize],
    ) -> Result<Box<dyn DeviceBuffer>, ferro_core::Error> {
        let f = |x: f32, y: f32| match kind {
            BinaryKind::Add => x + y,
            BinaryKind::Sub => x - y,
            BinaryKind::Mul => x * y,
            BinaryKind::Div => x / y,
        };
        let n: usize = out_shape.iter().product();
        let idx = |flat: usize, shape: &[usize]| -> usize {
            let pad = out_shape.len() - shape.len();
            let mut strides = vec![0usize; out_shape.len()];
            let (mut acc, mut off) = (1usize, 0usize);
            for d in (0..out_shape.len()).rev() {
                strides[d] = acc;
                acc *= out_shape[d];
            }
            for d in 0..out_shape.len() {
                let coord = (flat / strides[d]) % out_shape[d];
                if d >= pad && shape[d - pad] != 1 {
                    let mut s = 1usize;
                    for dd in (d - pad + 1)..shape.len() {
                        s *= shape[dd];
                    }
                    off += coord * s;
                }
            }
            off
        };
        let (va, vb) = (f32_data(a), f32_data(b));
        let out = (0..n).map(|i| f(va[idx(i, sa)], vb[idx(i, sb)])).collect();
        Ok(Box::new(F32Buf(out)))
    }
    fn gather_rows_dev(
        &self,
        w: &dyn DeviceBuffer,
        idx: &dyn DeviceBuffer,
        _dim_size: usize,
        inner: usize,
    ) -> Result<Box<dyn DeviceBuffer>, ferro_core::Error> {
        GATHER_ROWS.fetch_add(1, Ordering::SeqCst);
        let wt = f32_data(w);
        let ids = i64_data(idx);
        let mut out = vec![0f32; ids.len() * inner];
        for (o, &id) in ids.iter().enumerate() {
            let src = (id as usize) * inner;
            out[o * inner..(o + 1) * inner].copy_from_slice(&wt[src..src + inner]);
        }
        Ok(Box::new(F32Buf(out)))
    }
}

#[derive(Clone, Copy)]
struct Counts {
    f32_allocs: usize,
    i64_allocs: usize,
    to_host_f32_elems: usize,
    to_host_i64_elems: usize,
    gather_rows: usize,
}

fn counts() -> Counts {
    Counts {
        f32_allocs: F32_ALLOCS.load(Ordering::SeqCst),
        i64_allocs: I64_ALLOCS.load(Ordering::SeqCst),
        to_host_f32_elems: TO_HOST_F32_ELEMS.load(Ordering::SeqCst),
        to_host_i64_elems: TO_HOST_I64_ELEMS.load(Ordering::SeqCst),
        gather_rows: GATHER_ROWS.load(Ordering::SeqCst),
    }
}

static SERIAL: Mutex<()> = Mutex::new(());

fn setup() -> MutexGuard<'static, ()> {
    register_backend(DEV, Arc::new(CountingDevice));
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn index_select_backward_keeps_device_weight_on_device() {
    let _serial = setup();
    let wd = Tensor::from_vec(vec![0.3, -1.2, 0.7, 0.4, -0.5, 0.9], &[3, 2])
        .unwrap()
        .to_device(DEV)
        .unwrap()
        .requires_grad_(true)
        .unwrap();
    let idd = Tensor::from_vec_i64(vec![2, 0, 2], &[3])
        .unwrap()
        .to_device(DEV)
        .unwrap();
    let coefd = Tensor::from_vec(vec![0.4, 0.6, -0.2, 0.5, 0.9, -0.7], &[3, 2])
        .unwrap()
        .to_device(DEV)
        .unwrap();

    let before = counts();
    let out = wd.index_select_t(0, &idd).unwrap();
    assert_eq!(out.device(), DEV);
    let loss = out.mul(&coefd).unwrap().sum();
    assert_eq!(loss.device(), DEV);
    loss.backward();
    let g = wd.grad().expect("weight must receive a grad");
    assert_eq!(g.device(), DEV);
    let after = counts();

    // Exactly: one device gather kernel; the indices were read back once for
    // validation (never again - backward captured them); backward's only
    // upload is the scattered weight gradient; its only download is the one
    // cotangent read of the [3,2] upstream grad.
    assert_eq!(
        after.gather_rows - before.gather_rows,
        1,
        "exactly one gather kernel"
    );
    assert_eq!(
        after.to_host_i64_elems - before.to_host_i64_elems,
        3,
        "indices validated once"
    );
    assert_eq!(
        after.f32_allocs - before.f32_allocs,
        1,
        "only upload is the weight gradient"
    );
    assert_eq!(
        after.i64_allocs - before.i64_allocs,
        0,
        "no index re-uploads"
    );
    assert_eq!(
        after.to_host_f32_elems - before.to_host_f32_elems,
        6,
        "one cotangent read during backward"
    );

    // Values match the hand-computed duplicate accumulation (w row 0 <- C[1],
    // w row 2 <- C[0] + C[2]), and reading the grad costs exactly one
    // download of its size.
    let want = [-0.2f32, 0.5, 0.0, 0.0, 1.3, -0.1];
    for (got, e) in g.to_vec().iter().zip(want) {
        assert!((got - e).abs() < 1e-5, "grad {got} vs expected {e}");
    }
    let r0 = counts();
    let _ = g.to_vec();
    let r1 = counts();
    assert_eq!(r1.to_host_f32_elems - r0.to_host_f32_elems, 6);
}
