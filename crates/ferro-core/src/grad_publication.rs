//! Ordinary .grad publication; functional cotangents retain their own handles.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use super::{Tensor, Storage, Device, DType, Error, Result, backend_for, default_strides, raw_binary_k};
use crate::dispatch::BinaryKind;

const OP: &str = "gradient publication";

fn invalid(message: &str) -> Error { Error::Unsupported { op: OP, msg: message.into() } }

fn storage_key(value: &Tensor) -> usize { Arc::as_ptr(&value.0.storage) as usize }

fn whole_storage(value: &Tensor) -> Tensor {
    Tensor::from_parts(value.0.storage.clone(), vec![value.storage_len()], vec![1], 0, value.device(), false, None)
}

fn check_gradient(value: &Tensor, target: &Tensor) -> Result<()> {
    if value.shape() != target.shape() { return Err(Error::ShapeMismatch { op: OP, lhs: value.shape().to_vec(), rhs: target.shape().to_vec() }); }
    if value.dtype() != target.dtype() { return Err(Error::DtypeMismatch { op: OP, expected: target.dtype(), got: value.dtype() }); }
    if value.device() != target.device() { return Err(Error::DeviceMismatch { op: OP, lhs: target.device(), rhs: value.device() }); }
    Ok(())
}

fn layout_matches(gradient: &Tensor, target: &Tensor) -> bool {
    if target.has_dense_layout() {
        gradient.shape().iter().zip(gradient.strides()).zip(target.strides())
            .all(|((&size, &actual), &expected)| size == 1 || actual == expected)
    } else { gradient.is_contiguous() }
}

fn materialize_layout(value: &Tensor) -> Result<Option<Tensor>> {
    if value.device() == Device::Cpu || value.device_resident_whole() { return Ok(None); }
    let backend = backend_for(value.device())?;
    let storage = value.0.storage.read();
    let Storage::Device(buffer) = &*storage else { return Err(invalid("device gradients require float32 storage")); };
    match backend.materialize_support_for(buffer.as_ref(), value.shape(), value.strides(), value.storage_offset())? {
        crate::dispatch::MaterializeSupport::Attempt => {
            match backend.materialize_dev(buffer.as_ref(), value.shape(), value.strides(), value.storage_offset()) {
                Ok(buffer) => Ok(Some(super::device_leaf(buffer, value.shape(), value.device()))),
                Err(Error::MaterializeUnavailable) => Ok(None),
                Err(error) => Err(error),
            }
        }
        crate::dispatch::MaterializeSupport::UnsupportedBackend | crate::dispatch::MaterializeSupport::UnsupportedLayout => Ok(None),
    }
}

fn copy_layout(gradient: &Tensor, target: &Tensor) -> Result<Tensor> {
    let stride = if target.has_dense_layout() { target.strides().to_vec() } else { default_strides(target.shape()) };
    let mut axes = (0..target.ndim()).collect::<Vec<_>>();
    axes.sort_unstable_by_key(|&axis| std::cmp::Reverse(stride[axis]));
    let physical = Tensor::from_parts(gradient.0.storage.clone(), axes.iter().map(|&axis| gradient.shape()[axis]).collect(),
        axes.iter().map(|&axis| gradient.strides()[axis]).collect(), gradient.storage_offset(), gradient.device(), false, None);
    let owned = match materialize_layout(&physical)? {
        Some(resident) => resident,
        None => physical.try_owned_detach_copy()?.to_device(target.device())?,
    };
    let result = Tensor::from_parts(owned.0.storage.clone(), target.shape().to_vec(), stride, 0, target.device(), false, None);
    if target.has_dense_layout() { result.bump_version(); }
    Ok(result)
}

fn offsets(value: &Tensor) -> Result<Vec<usize>> {
    let count = value.numel();
    let mut indices = Vec::new();
    indices.try_reserve_exact(count).map_err(|_| invalid("gradient layout exceeds available capacity"))?;
    let mut seen = HashSet::new();
    seen.try_reserve(count).map_err(|_| invalid("gradient layout exceeds available capacity"))?;
    for index in 0..count {
        let mut remainder = index;
        let mut offset = value.storage_offset();
        for axis in (0..value.ndim()).rev() {
            offset = offset.checked_add((remainder % value.shape()[axis]).checked_mul(value.strides()[axis])
                .ok_or_else(|| invalid("gradient layout offset overflow"))?).ok_or_else(|| invalid("gradient layout offset overflow"))?;
            remainder /= value.shape()[axis];
        }
        if offset >= value.storage_len() { return Err(invalid("gradient layout exceeds its storage")); }
        if !seen.insert(offset) { return Err(invalid("an existing gradient has overlapping writable elements")); }
        indices.push(offset);
    }
    Ok(indices)
}

fn host(value: &Tensor) -> Result<Tensor> {
    if value.device() == Device::Cpu { return Ok(value.clone()); }
    let backend = backend_for(value.device())?;
    let data = {
        let storage = value.0.storage.read();
        match &*storage {
            Storage::Device(buffer) => backend.copy_to_host(buffer.as_ref())?,
            _ => return Err(invalid("device gradient storage must contain float32 elements")),
        }
    };
    Tensor::from_vec(value.gather_view(&data), value.shape())
}

struct Update { destination: Tensor, value: Tensor, additions: usize }

#[derive(Default)]
struct PendingStorage { index: HashMap<usize, usize>, updates: Vec<Update> }

impl PendingStorage {
    fn read(&self, original: &Tensor) -> Tensor {
        match self.index.get(&storage_key(original)) {
            Some(&index) => {
                let value = &self.updates[index].value;
                Tensor::from_parts(value.0.storage.clone(), original.shape().to_vec(), original.strides().to_vec(),
                    original.storage_offset(), value.device(), false, None)
            }
            None => original.detached_view(),
        }
    }

    fn add(&mut self, destination: &Tensor, gradient: &Tensor) -> Result<()> {
        if destination.is_structural_zero() { return Err(invalid("structural zero gradient storage is immutable; assign a materialized clone before accumulation")); }
        let positions = offsets(destination)?;
        let current = self.read(destination);
        let incoming = self.read(gradient);
        let incoming = if current.device_resident_whole() {
            materialize_layout(&incoming)?.unwrap_or(incoming)
        } else { incoming };
        let updated = if current.device_resident_whole() && incoming.device_resident_whole() {
            raw_binary_k(OP, &current, &incoming, BinaryKind::Add)?
        } else {
            let storage = self.read(&whole_storage(destination));
            let storage = host(&storage)?;
            let incoming = host(&incoming)?;
            match destination.dtype() {
                DType::F32 => {
                    let mut values = storage.to_vec();
                    for (offset, increment) in positions.into_iter().zip(incoming.to_vec()) { values[offset] += increment; }
                    Tensor::from_vec(values, &[destination.storage_len()])?
                }
                DType::F64 => {
                    let mut values = storage.to_vec_f64();
                    for (offset, increment) in positions.into_iter().zip(incoming.to_vec_f64()) { values[offset] += increment; }
                    Tensor::from_vec_f64(values, &[destination.storage_len()])?
                }
                _ => return Err(invalid("only floating gradients can accumulate")),
            }
        };
        let updated = whole_storage(&updated);
        let key = storage_key(destination);
        if let Some(&index) = self.index.get(&key) {
            let update = &mut self.updates[index];
            update.additions = update.additions.checked_add(1).ok_or_else(|| invalid("gradient update count overflow"))?;
            update.value = updated;
        } else {
            self.index.insert(key, self.updates.len());
            self.updates.push(Update { destination: whole_storage(destination), value: updated, additions: 1 });
        }
        Ok(())
    }

    fn commit(self) -> Result<()> {
        for update in self.updates {
            crate::inplace::raw_copy_(OP, &update.destination, &update.value)?;
            for _ in 1..update.additions { update.destination.bump_version(); }
        }
        Ok(())
    }
}

enum Slot { Keep, Set(Option<Tensor>) }

impl Tensor {
    pub(crate) fn publish_gradients(topo: &[Tensor], mut pending: HashMap<usize, Tensor>) -> Result<()> {
        let mut ordered = topo.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|value| value.id());
        let mut guards = ordered.iter().map(|value| value.0.grad.lock().unwrap_or_else(std::sync::PoisonError::into_inner)).collect::<Vec<_>>();
        // Separate leaves can expose views of the same manually assigned gradient.
        // Lock cells after slots, in address order, without holding any value lock.
        let mut cells = pending.values().chain(guards.iter().filter_map(|slot| slot.as_ref()))
            .map(|value| value.0.storage.clone()).collect::<Vec<_>>();
        cells.sort_unstable_by_key(|cell| Arc::as_ptr(cell) as usize);
        cells.dedup_by(|a, b| Arc::ptr_eq(a, b));
        let _publications = cells.iter().map(|cell| cell.publication.lock().unwrap_or_else(std::sync::PoisonError::into_inner)).collect::<Vec<_>>();
        let indices = ordered.iter().enumerate().map(|(index, value)| (value.id(), index)).collect::<HashMap<_, _>>();
        let mut slots = (0..ordered.len()).map(|_| Slot::Keep).collect::<Vec<_>>();
        let mut storage = PendingStorage::default();
        for target in topo {
            let index = indices[&target.id()];
            if target.0.op.is_some() { slots[index] = Slot::Set(None); }
            let Some(gradient) = pending.remove(&target.id()) else { continue };
            check_gradient(&gradient, target)?;
            if target.0.op.is_none() {
                if let Some(previous) = guards[index].as_ref() {
                    check_gradient(previous, target)?;
                    storage.add(previous, &gradient)?;
                    continue;
                }
            }
            let unique = Arc::strong_count(&gradient.0) == 1;
            let value = if unique && layout_matches(&gradient, target) {
                gradient.detached_view()
            } else { copy_layout(&storage.read(&gradient), target)? };
            slots[index] = Slot::Set(Some(value));
        }
        storage.commit()?;
        for (guard, slot) in guards.iter_mut().zip(slots) {
            if let Slot::Set(value) = slot { **guard = value; }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{is_grad_enabled, with_grad_enabled};

    fn value(shape: &[usize], dtype: DType) -> Tensor {
        Tensor::from_vec_f64(vec![2.0; shape.iter().product()], shape).unwrap().to_dtype(dtype)
    }

    #[test]
    fn direct_and_computed_live_callbacks_do_not_retain_published_ancestors() {
        for dtype in [DType::F32, DType::F64] { for shape in [&[][..], &[0][..], &[2, 3][..], &[0, 3][..]] { for computed in [false, true] {
            let x = value(shape, dtype).requires_grad_(true).unwrap();
            let weak = Arc::downgrade(&x.0);
            let parent = x.clone();
            let output = value(&[], dtype).record_fn(vec![x.clone()], move |_| {
                vec![if computed { with_grad_enabled(true, || parent.add(&parent).unwrap()) } else { parent.clone() }]
            });
            let weak_output = Arc::downgrade(&output.0);
            output.backward();
            let gradient = x.grad().unwrap();
            assert!(!gradient.requires_grad() && gradient.is_leaf());
            assert_ne!(gradient.id(), x.id());
            drop(output); drop(x);
            assert!(weak_output.upgrade().is_none());
            assert!(weak.upgrade().is_none(), "public gradient retained its input ancestor");
            assert_eq!(gradient.to_vec_f64(), vec![if computed { 4.0 } else { 2.0 }; shape.iter().product()]);
        } } }
    }

    #[test]
    fn efficient_zero_initial_slots_preserve_only_layout_compatible_markers() {
        for dtype in [DType::F32, DType::F64] { for shape in [&[][..], &[0][..], &[2, 3][..], &[0, 3][..], &[2, 0][..]] { for transpose in [false, true] {
            if transpose && shape.len() != 2 { continue; }
            let base = value(shape, dtype);
            let x = if transpose { base.transpose(0, 1).unwrap() } else { base };
            let x = x.detach().requires_grad_(true).unwrap();
            let dimensions = x.shape().to_vec();
            let output = value(&[], dtype).record_fn(vec![x.clone()], move |_| {
                vec![Tensor::structural_zero(&dimensions, dtype, Device::Cpu).unwrap()]
            });
            output.backward();
            let gradient = x.grad().unwrap();
            assert_eq!(gradient.is_structural_zero(), !transpose, "{dtype:?} {shape:?} transpose={transpose}");
            assert_eq!(gradient.strides(), x.strides());
            assert!(!gradient.requires_grad());
        } } }
    }

    #[test]
    fn immutable_existing_zero_rejects_before_other_slots_and_locks_remain_usable() {
        for dtype in [DType::F32, DType::F64] { for shape in [&[][..], &[0][..], &[2][..]] {
            let a = value(shape, dtype).requires_grad_(true).unwrap();
            let b = value(shape, dtype).requires_grad_(true).unwrap();
            let ordinary = value(shape, dtype);
            let zero = Tensor::structural_zero(shape, dtype, Device::Cpu).unwrap();
            a.set_leaf_grad(Some(ordinary.clone())).unwrap(); b.set_leaf_grad(Some(zero.clone())).unwrap();
            let before = ordinary.to_vec_f64();
            let loss = a.add(&b).unwrap().sum();
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loss.backward())).is_err());
            assert_eq!(ordinary.to_vec_f64(), before); assert_eq!(ordinary._version(), 0);
            assert_eq!(a.grad().unwrap().id(), ordinary.id()); assert_eq!(b.grad().unwrap().id(), zero.id());
            assert!(is_grad_enabled());
            b.set_leaf_grad(Some(zero.clone_tensor(true).unwrap())).unwrap(); loss.backward();
            assert_eq!(ordinary.to_vec_f64(), vec![3.0; shape.iter().product()]);
        } }
    }

    #[test]
    fn poisoned_publication_mutex_recovers_without_losing_values() {
        let x = value(&[2], DType::F32).requires_grad_(true).unwrap();
        let gradient = value(&[2], DType::F32);
        x.set_leaf_grad(Some(gradient.clone())).unwrap();
        let cell = gradient.0.storage.clone();
        assert!(std::thread::spawn(move || { let _guard = cell.publication.lock().unwrap(); panic!("poison test"); }).join().is_err());
        x.sum().backward();
        assert_eq!(x.grad().unwrap().id(), gradient.id()); assert_eq!(gradient.to_vec(), vec![3.0; 2]);
    }
}
