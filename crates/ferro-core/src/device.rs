use std::fmt;

/// Where a tensor's storage lives. Cpu is the only device with kernels today;
/// the Cuda variant exists so device plumbing (creation, equality checks,
/// dispatch) is exercised now and a real backend crate can slot in later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(u32),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
        }
    }
}
