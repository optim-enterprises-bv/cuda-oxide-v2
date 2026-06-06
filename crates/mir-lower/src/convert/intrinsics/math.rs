/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Math approximation intrinsic conversion.
//!
//! | Operation         | Lowering                  | PTX Output         |
//! |-------------------|---------------------------|--------------------|
//! | `SqrtApproxF32Op` | `llvm_nvvm_sqrt_approx_f` | `sqrt.approx.f32`  |
//! | `CosApproxF32Op`  | `llvm_nvvm_cos_approx_f`  | `cos.approx.f32`   |
//! | `SinApproxF32Op`  | `llvm_nvvm_sin_approx_f`  | `sin.approx.f32`   |
//! | `Ex2ApproxF32Op`  | `llvm_nvvm_ex2_approx_f`  | `ex2.approx.f32`   |
//! | `Lg2ApproxF32Op`  | `llvm_nvvm_lg2_approx_f`  | `lg2.approx.f32`   |

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::builtin::types::FP32Type;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert a unary `*.approx.f32` math op to the matching LLVM NVVM
/// intrinsic call. All five members of the family (`sqrt`, `cos`,
/// `sin`, `ex2`, `lg2`) share this body — the op type and
/// `intrinsic_name` distinguish them.
pub(crate) fn convert_unary_approx_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let f32_ty = FP32Type::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!(
            "*.approx.f32 unary math op requires 1 operand, got {}",
            operands.len()
        );
    }
    let x = operands[0];

    let func_ty = llvm_types::FuncType::get(ctx, f32_ty.into(), vec![f32_ty.into()], false);

    let call_op = call_intrinsic(ctx, rewriter, op, intrinsic_name, func_ty, vec![x])?;
    rewriter.replace_operation(ctx, op, call_op);

    Ok(())
}
