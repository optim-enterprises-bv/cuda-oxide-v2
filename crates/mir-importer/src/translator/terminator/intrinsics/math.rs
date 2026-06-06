/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Math approximation intrinsics (PTX `*.approx.f32` family).
//!
//! Translates calls to `cuda_device::math::{sqrt,cos,sin,ex2,lg2}_
//! approx_f32` into the matching `dialect_nvvm` op. The lowering pass
//! later replaces the op with an `llvm.nvvm.*` intrinsic call in the
//! emitted LLVM IR; the NVPTX backend then emits a single PTX
//! `*.approx.f32` instruction.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::FP32Type;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit one of the unary `*.approx.f32` math intrinsics. Pass the
/// dialect-nvvm op's concrete op info — same shape for all five ops in
/// the family (1 f32 operand, 1 f32 result), so they share this body.
#[allow(clippy::too_many_arguments)]
pub fn emit_unary_approx_f32(
    ctx: &mut Context,
    body: &mir::Body,
    op_info: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    op_label: &'static str,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{op_label} expects 1 argument (x: f32), got {}",
                args.len()
            ))
        );
    }

    let f32_type = FP32Type::get(ctx);

    let (x_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let math_op = Operation::new(ctx, op_info, vec![f32_type.into()], vec![x_val], vec![], 0);
    math_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        math_op.insert_after(ctx, prev);
    } else {
        math_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = math_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        math_op,
        value_map,
        block_map,
        loc,
        "math approx call without target block",
    )
}
