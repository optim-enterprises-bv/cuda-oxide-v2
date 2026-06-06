/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Math approximation intrinsics (PTX `*.approx.f32` family).
//!
//! Each op corresponds to a single LLVM NVPTX intrinsic that the
//! backend lowers directly to a single `sqrt.approx.f32` /
//! `cos.approx.f32` / etc PTX instruction. No libdevice / libNVVM
//! dependency. ulp budgets are as documented in the PTX ISA.
//!
//! ```text
//! ┌──────────────────────┬───────────────────────────┬────────────────────┐
//! │ Operation            │ LLVM Intrinsic            │ PTX                │
//! ├──────────────────────┼───────────────────────────┼────────────────────┤
//! │ SqrtApproxF32Op      │ llvm.nvvm.sqrt.approx.f   │ sqrt.approx.f32    │
//! │ CosApproxF32Op       │ llvm.nvvm.cos.approx.f    │ cos.approx.f32     │
//! │ SinApproxF32Op       │ llvm.nvvm.sin.approx.f    │ sin.approx.f32     │
//! │ Ex2ApproxF32Op       │ llvm.nvvm.ex2.approx.f    │ ex2.approx.f32     │
//! │ Lg2ApproxF32Op       │ llvm.nvvm.lg2.approx.f    │ lg2.approx.f32     │
//! └──────────────────────┴───────────────────────────┴────────────────────┘
//! ```

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Approximate sqrt for f32.
///
/// Corresponds to `llvm.nvvm.sqrt.approx.f` / PTX `sqrt.approx.f32`.
#[pliron_op(
    name = "nvvm.sqrt_approx_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct SqrtApproxF32Op;

impl SqrtApproxF32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        SqrtApproxF32Op { op }
    }
}

/// Approximate cosine for f32.
///
/// Corresponds to `llvm.nvvm.cos.approx.f` / PTX `cos.approx.f32`.
#[pliron_op(
    name = "nvvm.cos_approx_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct CosApproxF32Op;

impl CosApproxF32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        CosApproxF32Op { op }
    }
}

/// Approximate sine for f32.
///
/// Corresponds to `llvm.nvvm.sin.approx.f` / PTX `sin.approx.f32`.
#[pliron_op(
    name = "nvvm.sin_approx_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct SinApproxF32Op;

impl SinApproxF32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        SinApproxF32Op { op }
    }
}

/// Approximate 2^x for f32.
///
/// Corresponds to `llvm.nvvm.ex2.approx.f` / PTX `ex2.approx.f32`.
#[pliron_op(
    name = "nvvm.ex2_approx_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct Ex2ApproxF32Op;

impl Ex2ApproxF32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Ex2ApproxF32Op { op }
    }
}

/// Approximate log2(x) for f32.
///
/// Corresponds to `llvm.nvvm.lg2.approx.f` / PTX `lg2.approx.f32`.
#[pliron_op(
    name = "nvvm.lg2_approx_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct Lg2ApproxF32Op;

impl Lg2ApproxF32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Lg2ApproxF32Op { op }
    }
}

/// Register math operations with the context.
pub(super) fn register(ctx: &mut Context) {
    SqrtApproxF32Op::register(ctx);
    CosApproxF32Op::register(ctx);
    SinApproxF32Op::register(ctx);
    Ex2ApproxF32Op::register(ctx);
    Lg2ApproxF32Op::register(ctx);
}
