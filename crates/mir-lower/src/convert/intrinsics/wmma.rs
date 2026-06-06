/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (per-warp 16x16x16 tensor-core MMA) intrinsic conversion for Ampere+.
//!
//! Lowering strategy: inline PTX assembly. LLVM's `llvm.nvvm.wmma.*`
//! intrinsics route through NVVM-IR (libNVVM) which our pipeline does
//! not use — direct inline PTX produces clean output through `llc`.
//!
//! | Operation                              | PTX                                                                    |
//! |----------------------------------------|------------------------------------------------------------------------|
//! | `WmmaLoadAM16N16K16Bf16RowOp`          | `wmma.load.a.sync.aligned.m16n16k16.row.bf16`                          |
//! | `WmmaLoadBM16N16K16Bf16ColOp`          | `wmma.load.b.sync.aligned.m16n16k16.col.bf16`                          |
//! | `WmmaMmaM16N16K16Bf16Bf16F32Op`        | `wmma.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32`            |
//! | `WmmaStoreDM16N16K16Bf16F32RowOp`      | `wmma.store.d.sync.aligned.m16n16k16.row.f32`                          |

use llvm_export::ops as llvm;
use llvm_export::ops::InlineAsmOpExt;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeObj;

/// Common body for the two WMMA load shapes (load.a row, load.b col).
/// Both produce 4 i32 results out of a `{ptr, stride}` pair.
fn convert_load_4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    asm_template: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!(
            "wmma load requires 2 operands [ptr, stride], got {}",
            operands.len()
        );
    }
    let ptr = operands[0];
    let stride = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let field_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| i32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        vec![ptr, stride],
        asm_template,
        "=r,=r,=r,=r,l,r",
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);

    let struct_result = asm_op.deref(ctx).get_result(0);
    let mut extracted_values = Vec::with_capacity(4);
    for i in 0..4u32 {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![i])
            .map_err(|e| pliron::input_error_noloc!("{}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let field_val = extract_op.get_operation().deref(ctx).get_result(0);
        extracted_values.push(field_val);
    }
    rewriter.replace_operation_with_values(ctx, op, extracted_values);
    Ok(())
}

pub(crate) fn convert_load_a_m16n16k16_bf16_row(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_load_4(
        ctx,
        rewriter,
        op,
        "wmma.load.a.sync.aligned.m16n16k16.row.bf16 {$0,$1,$2,$3}, [$4], $5;",
    )
}

pub(crate) fn convert_load_b_m16n16k16_bf16_col(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_load_4(
        ctx,
        rewriter,
        op,
        "wmma.load.b.sync.aligned.m16n16k16.col.bf16 {$0,$1,$2,$3}, [$4], $5;",
    )
}

pub(crate) fn convert_mma_m16n16k16_bf16_bf16_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 16 {
        return pliron::input_err_noloc!(
            "wmma mma requires 16 operands [a0..3, b0..3, c0..7], got {}",
            operands.len()
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let field_types: Vec<Ptr<TypeObj>> = (0..8).map(|_| f32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let asm_template = "wmma.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 \
{$0,$1,$2,$3,$4,$5,$6,$7}, \
{$8,$9,$10,$11}, \
{$12,$13,$14,$15}, \
{$16,$17,$18,$19,$20,$21,$22,$23};";

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        operands,
        asm_template,
        "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,r,r,r,r,f,f,f,f,f,f,f,f",
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);

    let struct_result = asm_op.deref(ctx).get_result(0);
    let mut extracted_values = Vec::with_capacity(8);
    for i in 0..8u32 {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![i])
            .map_err(|e| pliron::input_error_noloc!("{}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let field_val = extract_op.get_operation().deref(ctx).get_result(0);
        extracted_values.push(field_val);
    }
    rewriter.replace_operation_with_values(ctx, op, extracted_values);
    Ok(())
}

pub(crate) fn convert_store_d_m16n16k16_bf16_f32_row(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "wmma store_d requires 10 operands [dst, stride, d0..7], got {}",
            operands.len()
        );
    }

    let void_ty = llvm_types::VoidType::get(ctx);
    let asm_template = "wmma.store.d.sync.aligned.m16n16k16.row.f32 \
[$0], \
{$2,$3,$4,$5,$6,$7,$8,$9}, $1;";

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        void_ty.into(),
        operands,
        asm_template,
        "l,r,f,f,f,f,f,f,f,f,~{memory}",
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.erase_operation(ctx, op);
    Ok(())
}
