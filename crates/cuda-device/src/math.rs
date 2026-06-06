//! PTX-direct math approximation intrinsics.
//!
//! Each function in this module lowers to a single `*.approx.f32` PTX
//! instruction via the corresponding `llvm.nvvm.*.approx.f` intrinsic.
//! Faster than libdevice (`__nv_sqrtf`, `__nv_cosf`, ...) but with
//! looser precision — within the PTX ISA-documented ulp budget.
//!
//! These stubs deliberately have no host-side implementation; calling
//! them outside a `#[kernel]` context panics. The `mir-importer` pass
//! recognises the fully-qualified path and replaces the call with a
//! `dialect_nvvm` op, which `mir-lower` then converts to an
//! `llvm.nvvm.*` intrinsic call in the emitted LLVM IR.

/// Square root, approximate. Lowers to PTX `sqrt.approx.f32` via
/// `llvm.nvvm.sqrt.approx.f`.
#[inline(never)]
pub fn sqrt_approx_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!("sqrt_approx_f32 called outside CUDA kernel context")
}

/// Cosine, approximate. Lowers to PTX `cos.approx.f32` via
/// `llvm.nvvm.cos.approx.f`.
#[inline(never)]
pub fn cos_approx_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!("cos_approx_f32 called outside CUDA kernel context")
}

/// Sine, approximate. Lowers to PTX `sin.approx.f32` via
/// `llvm.nvvm.sin.approx.f`.
#[inline(never)]
pub fn sin_approx_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!("sin_approx_f32 called outside CUDA kernel context")
}

/// 2^x, approximate. Lowers to PTX `ex2.approx.f32` via
/// `llvm.nvvm.ex2.approx.f`.
#[inline(never)]
pub fn ex2_approx_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!("ex2_approx_f32 called outside CUDA kernel context")
}

/// log2(x), approximate. Lowers to PTX `lg2.approx.f32` via
/// `llvm.nvvm.lg2.approx.f`.
#[inline(never)]
pub fn lg2_approx_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!("lg2_approx_f32 called outside CUDA kernel context")
}
