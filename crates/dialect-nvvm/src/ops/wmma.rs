//! Per-warp WMMA tensor-core operations for Ampere (sm_80+).
//!
//! 16x16x16 matrix multiply-accumulate distributed across a warp (32
//! threads). Inputs are bf16, accumulator is f32. Each thread holds a
//! sub-fragment of the tile in registers.
//!
//! | Operation                              | PTX                                                                    |
//! |----------------------------------------|------------------------------------------------------------------------|
//! | `WmmaLoadAM16N16K16Bf16RowOp`          | `wmma.load.a.sync.aligned.m16n16k16.row.bf16`                          |
//! | `WmmaLoadBM16N16K16Bf16ColOp`          | `wmma.load.b.sync.aligned.m16n16k16.col.bf16`                          |
//! | `WmmaMmaM16N16K16Bf16Bf16F32Op`        | `wmma.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32`            |
//! | `WmmaStoreDM16N16K16Bf16F32RowOp`      | `wmma.store.d.sync.aligned.m16n16k16.row.f32`                          |

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Warp-cooperative load of A fragment. m16n16k16 bf16, row-major.
///
/// Operands: `[src_ptr, stride]`. Results: 4 u32 (packed bf16 pairs).
#[pliron_op(
    name = "nvvm.wmma_load_a_m16n16k16_bf16_row",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<4>],
)]
pub struct WmmaLoadAM16N16K16Bf16RowOp;

impl WmmaLoadAM16N16K16Bf16RowOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        WmmaLoadAM16N16K16Bf16RowOp { op }
    }
}

/// Warp-cooperative load of B fragment. m16n16k16 bf16, column-major.
///
/// Operands: `[src_ptr, stride]`. Results: 4 u32 (packed bf16 pairs).
#[pliron_op(
    name = "nvvm.wmma_load_b_m16n16k16_bf16_col",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<4>],
)]
pub struct WmmaLoadBM16N16K16Bf16ColOp;

impl WmmaLoadBM16N16K16Bf16ColOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        WmmaLoadBM16N16K16Bf16ColOp { op }
    }
}

/// Warp-cooperative MMA: `D = A * B + C`. m16n16k16 bf16xbf16 -> f32.
///
/// Operands: `[a0..a3, b0..b3, c0..c7]` (4 u32 + 4 u32 + 8 f32 = 16).
/// Results: 8 f32 (the new accumulator lanes).
#[pliron_op(
    name = "nvvm.wmma_mma_m16n16k16_bf16_bf16_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<16>, NResultsInterface<8>],
)]
pub struct WmmaMmaM16N16K16Bf16Bf16F32Op;

impl WmmaMmaM16N16K16Bf16Bf16F32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        WmmaMmaM16N16K16Bf16Bf16F32Op { op }
    }
}

/// Warp-cooperative store of D fragment. m16n16k16 f32, row-major.
///
/// Operands: `[dst_ptr, stride, d0..d7]` (1 ptr + 1 u32 + 8 f32 = 10).
/// Results: none.
#[pliron_op(
    name = "nvvm.wmma_store_d_m16n16k16_bf16_f32_row",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<10>, NResultsInterface<0>],
)]
pub struct WmmaStoreDM16N16K16Bf16F32RowOp;

impl WmmaStoreDM16N16K16Bf16F32RowOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        WmmaStoreDM16N16K16Bf16F32RowOp { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    WmmaLoadAM16N16K16Bf16RowOp::register(ctx);
    WmmaLoadBM16N16K16Bf16ColOp::register(ctx);
    WmmaMmaM16N16K16Bf16Bf16F32Op::register(ctx);
    WmmaStoreDM16N16K16Bf16F32RowOp::register(ctx);
}
