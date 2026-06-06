/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (per-warp 16x16x16 tensor-core MMA) intrinsic emitters for Ampere+.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    WmmaLoadAM16N16K16Bf16RowOp, WmmaLoadBM16N16K16Bf16ColOp,
    WmmaMmaM16N16K16Bf16Bf16F32Op, WmmaStoreDM16N16K16Bf16F32RowOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeObj;
use pliron::value::Value;
use rustc_public::mir;

/// Translate the destination place type to its full-layout MIR struct
/// type. Required so the constructed `CuSimd` struct matches the alloca
/// slot emitted for the destination local.
fn destination_struct_type(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    loc: Location,
) -> TranslationResult<Ptr<TypeObj>> {
    let dest_rust_ty = match destination.ty(body.locals()) {
        Ok(t) => t,
        Err(e) => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "failed to resolve destination type for WMMA result: {e:?}"
                ))
            );
        }
    };
    types::translate_type(ctx, &dest_rust_ty)
}

/// Common load body for the two WMMA load shapes. Builds an op with
/// `[src, stride]` operands and 4 results of element type `elt_ty`,
/// then wraps the results into a `[elt; 4]` array and the destination's
/// `CuSimd<elt, 4>` struct.
#[allow(clippy::too_many_arguments)]
fn emit_wmma_load_4(
    ctx: &mut Context,
    body: &mir::Body,
    op_info: (
        fn(Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    op_label: &'static str,
    elt_ty: Ptr<TypeObj>,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{op_label} expects 2 arguments (src, stride), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(2);
    for arg in args.iter().take(2) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let result_types = (0..4).map(|_| elt_ty).collect();
    let ld_op = Operation::new(ctx, op_info, result_types, operands, vec![], 0);
    ld_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    let results: Vec<Value> = (0..4).map(|i| ld_op.deref(ctx).get_result(i)).collect();

    let array_ty = dialect_mir::types::MirArrayType::get(ctx, elt_ty, 4);
    let array_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    array_op.insert_after(ctx, ld_op);

    let struct_ty = destination_struct_type(ctx, body, destination, loc.clone())?;
    let array_result = array_op.deref(ctx).get_result(0);
    let struct_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructStructOp::get_concrete_op_info(),
        vec![struct_ty],
        vec![array_result],
        vec![],
        0,
    );
    struct_op.deref_mut(ctx).set_loc(loc.clone());
    struct_op.insert_after(ctx, array_op);

    let struct_result = struct_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        struct_result,
        target,
        block_ptr,
        struct_op,
        value_map,
        block_map,
        loc,
        op_label,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_wmma_load_a_m16n16k16_bf16_row(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let elt_ty = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    emit_wmma_load_4(
        ctx, body,
        WmmaLoadAM16N16K16Bf16RowOp::get_concrete_op_info(),
        "wmma_load_a_m16n16k16_bf16_row",
        elt_ty,
        args, destination, target, block_ptr, prev_op, value_map, block_map, loc,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_wmma_load_b_m16n16k16_bf16_col(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let elt_ty = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    emit_wmma_load_4(
        ctx, body,
        WmmaLoadBM16N16K16Bf16ColOp::get_concrete_op_info(),
        "wmma_load_b_m16n16k16_bf16_col",
        elt_ty,
        args, destination, target, block_ptr, prev_op, value_map, block_map, loc,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_wmma_mma_m16n16k16_bf16_bf16_f32_raw(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 16 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "wmma_mma_m16n16k16_bf16_bf16_f32_raw expects 16 arguments \
                 (a0..a3, b0..b3, c0..c7), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(16);
    for arg in args.iter().take(16) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let f32_ty = FP32Type::get(ctx);
    let result_types = (0..8).map(|_| f32_ty.into()).collect();

    let mma_op = Operation::new(
        ctx,
        WmmaMmaM16N16K16Bf16Bf16F32Op::get_concrete_op_info(),
        result_types,
        operands,
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    let results: Vec<Value> = (0..8).map(|i| mma_op.deref(ctx).get_result(i)).collect();

    let array_ty = dialect_mir::types::MirArrayType::get(ctx, f32_ty.into(), 8);
    let array_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    array_op.insert_after(ctx, mma_op);

    let struct_ty = destination_struct_type(ctx, body, destination, loc.clone())?;
    let array_result = array_op.deref(ctx).get_result(0);
    let struct_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructStructOp::get_concrete_op_info(),
        vec![struct_ty],
        vec![array_result],
        vec![],
        0,
    );
    struct_op.deref_mut(ctx).set_loc(loc.clone());
    struct_op.insert_after(ctx, array_op);

    let struct_result = struct_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        struct_result,
        target,
        block_ptr,
        struct_op,
        value_map,
        block_map,
        loc,
        "wmma_mma_m16n16k16_bf16_bf16_f32_raw call without target block",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_wmma_store_d_m16n16k16_bf16_f32_row_raw(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 10 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "wmma_store_d_m16n16k16_bf16_f32_row_raw expects 10 arguments \
                 (dst, stride, d0..d7), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(10);
    for arg in args.iter().take(10) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let st_op = Operation::new(
        ctx,
        WmmaStoreDM16N16K16Bf16F32RowOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, st_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "wmma_store_d_m16n16k16_bf16_f32_row_raw call without target block".to_string()
            )
        )
    }
}
