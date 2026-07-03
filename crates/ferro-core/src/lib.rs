//! ferro-core: a small, dependency-free PyTorch-style tensor + reverse-mode
//! autograd runtime in Rust. This is the authoritative compute/differentiation
//! layer; Python bindings and device backends live in sibling crates.

mod autograd;
pub mod device;
pub mod dispatch;
pub mod dtype;
pub mod error;
pub mod interop;
pub mod nn;
pub mod ops;
pub mod ops_ext;
pub mod optim;
pub mod params;
pub mod rng;
pub mod shape;
pub mod tensor;
pub mod testkit;

pub use device::Device;
pub use dispatch::{register_backend, Backend, BinaryKind, CpuBackend, UnaryKind};
pub use dtype::DType;
pub use error::{Error, Result};
pub use params::Param;
pub use rng::Rng;
pub use tensor::{Storage, Tensor};
