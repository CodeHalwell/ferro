//! ferro-core: a small, dependency-free PyTorch-style tensor + reverse-mode
//! autograd runtime in Rust. This is the authoritative compute/differentiation
//! layer; Python bindings and device backends live in sibling crates.

pub mod amp;
pub mod autograd;
pub mod checkpoint;
pub mod data;
pub mod ddp;
pub mod device;
pub mod dispatch;
pub mod dtype;
pub mod error;
pub mod fused_ops;
pub mod graph;
pub mod half;
pub mod inplace;
pub mod interop;
pub mod modules;
pub mod nn;
pub mod ops;
pub mod ops_ext;
pub mod optim;
pub mod params;
pub mod philox;
pub mod pool;
pub mod replay;
mod reduce;
pub mod rng;
pub mod safetensors;
pub mod shape;
pub mod tensor;
pub mod testkit;

pub use device::Device;
pub use dispatch::{
    register_backend, Backend, BinaryKind, ChainStepRef, CpuBackend, OpTag, UnaryKind,
};
pub use dtype::DType;
pub use error::{Error, Result};
pub use params::Param;
pub use rng::Rng;
pub use safetensors::{load_safetensors, save_safetensors};
pub use tensor::{Storage, Tensor};
