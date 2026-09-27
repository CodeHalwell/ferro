use ferro_core::dispatch::{backend_for, set_matmul_kernel, Backend};
use ferro_core::{CpuBackend, Device, Tensor};
use std::sync::{Arc, Mutex, MutexGuard};

static REGISTRY: Mutex<()> = Mutex::new(());

// Every unit test touching BACKENDS or MATMUL holds this for its full lifetime.
// Direct FastCpuBackend calls and packed arithmetic do not need this lock.
// MATMUL has no getter: all participants restore its process-start baseline.
// Backend identity is saved, rather than manufacturing a replacement Arc.
pub(crate) struct RegistryGuard {
    previous: Arc<dyn Backend>,
    _lock: MutexGuard<'static, ()>,
}

pub(crate) fn lock() -> RegistryGuard {
    let guard = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    RegistryGuard { previous: backend_for(Device::Cpu).unwrap(), _lock: guard }
}

impl Drop for RegistryGuard {
    fn drop(&mut self) {
        set_matmul_kernel(ferro_core::dispatch::naive_matmul);
        ferro_core::register_backend(Device::Cpu, self.previous.clone());
    }
}

#[test]
fn registry_scope_restores_after_unwind() {
    let previous = {
        let _guard = lock();
        backend_for(Device::Cpu).unwrap()
    };
    let panic = std::panic::catch_unwind(|| {
        let _guard = lock();
        assert!(matches!(REGISTRY.try_lock(), Err(std::sync::TryLockError::WouldBlock)));
        crate::install_backend();
        set_matmul_kernel(|_, _, m, _, n| vec![-123.0; m * n]);
        assert_eq!(CpuBackend.matmul(&[2.0], &[3.0], 1, 1, 1), vec![-123.0]);
        panic!("intentional registry-owner unwind");
    });
    assert!(panic.is_err());
    let _guard = lock();
    let restored = backend_for(Device::Cpu).unwrap();
    let a = Tensor::from_vec(vec![2.0], &[1, 1]).unwrap();
    let b = Tensor::from_vec(vec![3.0], &[1, 1]).unwrap();
    assert_eq!(
        (Arc::ptr_eq(&previous, &restored), CpuBackend.matmul(&[2.0], &[3.0], 1, 1, 1)),
        (true, vec![6.0]),
        "backend and MATMUL mutations must not escape the registry scope",
    );
    assert_eq!(a.matmul(&b).unwrap().to_vec(), vec![6.0]);
}
