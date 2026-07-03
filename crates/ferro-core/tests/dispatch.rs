//! Backend registry and named-kernel dispatch. Notes:
//! - The registry is process-global, so these tests never replace the Cpu
//!   backend (that would poison every other test in the binary); the routing
//!   test registers under a Cuda device instead.
//! - Tensors are still Cpu-only, so kind-routed ops are exercised end to end
//!   on Cpu and the registry is probed directly for other devices.

use std::sync::Arc;

use ferro_core::dispatch::backend_for;
use ferro_core::testkit::grad_check;
use ferro_core::{Backend, BinaryKind, CpuBackend, Device, Tensor, UnaryKind};

#[test]
fn unregistered_device_errors() {
    let err = backend_for(Device::Cuda(7)).err().expect("no cuda backend");
    let msg = err.to_string();
    assert!(msg.contains("no backend registered for device cuda:7"), "got: {msg}");
}

/// Deliberately wrong math so routing to it is unmistakable.
struct PlusThousand;

impl Backend for PlusThousand {
    fn unary(&self, _kind: UnaryKind, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v + 1000.0).collect()
    }
    fn binary(&self, _kind: BinaryKind, a: &[f32], _b: &[f32]) -> Vec<f32> {
        a.to_vec()
    }
    fn matmul(&self, a: &[f32], _b: &[f32], _m: usize, _k: usize, _n: usize) -> Vec<f32> {
        a.to_vec()
    }
}

#[test]
fn registry_routes_per_device() {
    ferro_core::register_backend(Device::Cuda(0), Arc::new(PlusThousand));
    let fake = backend_for(Device::Cuda(0)).unwrap();
    assert_eq!(fake.unary(UnaryKind::Neg, &[1.0, 2.0]), vec![1001.0, 1002.0]);
    // Cpu is untouched by the Cuda registration.
    let cpu = backend_for(Device::Cpu).unwrap();
    assert_eq!(cpu.unary(UnaryKind::Neg, &[1.0, 2.0]), vec![-1.0, -2.0]);
}

fn check_unary(kind: UnaryKind, x: &[f32], f: impl Fn(f32) -> f32) {
    let want: Vec<f32> = x.iter().map(|&v| f(v)).collect();
    assert_eq!(CpuBackend.unary(kind, x), want, "{kind:?}");
}

// The max/min chain is deliberate: it is the torch clamp semantics the
// backend implements, not f32::clamp (which panics on min > max).
#[allow(clippy::manual_clamp)]
#[test]
fn cpu_unary_kinds_match_reference() {
    let x = [-2.0f32, -0.5, 0.0, 0.75, 3.0];
    let pos = [0.25f32, 1.0, 2.5, 9.0];
    check_unary(UnaryKind::Neg, &x, |v| -v);
    check_unary(UnaryKind::Relu, &x, |v| v.max(0.0));
    check_unary(UnaryKind::Exp, &x, |v| v.exp());
    check_unary(UnaryKind::Sigmoid, &x, |v| 1.0 / (1.0 + (-v).exp()));
    check_unary(UnaryKind::Tanh, &x, |v| v.tanh());
    check_unary(UnaryKind::Sqrt, &pos, |v| v.sqrt());
    check_unary(UnaryKind::Abs, &x, |v| v.abs());
    check_unary(UnaryKind::Log, &pos, |v| v.ln());
    check_unary(UnaryKind::Powf(3.0), &x, |v| v.powf(3.0));
    check_unary(UnaryKind::Clamp { min: -1.0, max: 1.0 }, &x, |v| v.max(-1.0).min(1.0));
    // torch semantics when min > max: max everywhere, no panic.
    check_unary(UnaryKind::Clamp { min: 2.0, max: 1.0 }, &x, |_| 1.0);
}

#[test]
fn cpu_binary_kinds_match_reference() {
    let a = [1.5f32, -2.0, 0.0, 4.0];
    let b = [0.5f32, 3.0, -1.0, 2.0];
    let zip = || a.iter().zip(b.iter());
    assert_eq!(CpuBackend.binary(BinaryKind::Add, &a, &b), zip().map(|(&x, &y)| x + y).collect::<Vec<_>>());
    assert_eq!(CpuBackend.binary(BinaryKind::Sub, &a, &b), zip().map(|(&x, &y)| x - y).collect::<Vec<_>>());
    assert_eq!(CpuBackend.binary(BinaryKind::Mul, &a, &b), zip().map(|(&x, &y)| x * y).collect::<Vec<_>>());
    assert_eq!(CpuBackend.binary(BinaryKind::Div, &a, &b), zip().map(|(&x, &y)| x / y).collect::<Vec<_>>());
}

#[test]
fn cpu_matmul_matches_reference() {
    // (2,3) @ (3,2), hand-computed.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    assert_eq!(CpuBackend.matmul(&a, &b, 2, 3, 2), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn kind_routed_binary_broadcasts() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    assert_eq!(a.add(&b).unwrap().to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn kind_routed_tanh_grad_checks() {
    let x = Tensor::from_vec(vec![-1.5, -0.3, 0.0, 0.4, 1.2, 2.0], &[2, 3]).unwrap();
    grad_check(&[x], |leaves| leaves[0].tanh().sum());
}
