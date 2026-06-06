//! Per-warp WMMA tensor-core intrinsics for Ampere (sm_80+).
//!
//! WMMA performs a warp-cooperative 16x16x16 matrix multiply-accumulate:
//! `D[16,16] = A[16,16] * B[16,16] + C[16,16]`. Each warp (32 threads)
//! collectively holds the fragments — no one thread sees the full tile.
//!
//! # Per-thread fragment layout (m16n16k16, bf16 inputs / f32 accumulator)
//!
//! * `A` fragment: 4 x u32 per thread (each u32 packs two bf16 lanes).
//! * `B` fragment: 4 x u32 per thread, same packing.
//! * `C` / `D` accumulator: 8 x f32 per thread.
//!
//! Total per warp: 4*32 = 128 u32 = 256 bf16 = 16x16 tile (A and B);
//! 8*32 = 256 f32 = 16x16 tile (C and D).
//!
//! # Usage
//!
//! ```rust,ignore
//! use cuda_device::wmma::*;
//!
//! let a = unsafe { wmma_load_a_m16n16k16_bf16_row(a_smem_ptr, k_stride) };
//! let b = unsafe { wmma_load_b_m16n16k16_bf16_col(b_smem_ptr, k_stride) };
//! let mut acc = WmmaAccF32::new([0.0; 8]);
//! acc = unsafe { wmma_mma_m16n16k16_bf16_bf16_f32(a, b, acc) };
//! unsafe { wmma_store_d_m16n16k16_bf16_f32_row(dst, acc, n_stride) };
//! ```
//!
//! # Hardware support
//!
//! * `sm_80` (A100, A30)
//! * `sm_86` (RTX 3060, 3090, A10)
//! * `sm_89` (RTX 4090, L40)
//! * `sm_90+` (Hopper, Blackwell — also support more efficient `wgmma`)

use crate::cusimd::CuSimd;

/// Per-thread fragment of the A operand in m16n16k16 bf16 WMMA.
pub type WmmaFragABf16 = CuSimd<u32, 4>;

/// Per-thread fragment of the B operand in m16n16k16 bf16 WMMA.
pub type WmmaFragBBf16 = CuSimd<u32, 4>;

/// Per-thread f32 accumulator fragment for m16n16k16 WMMA.
pub type WmmaAccF32 = CuSimd<f32, 8>;

/// WMMA load of the A operand. m16n16k16, bf16 elements, row-major tile.
/// Lowers to PTX `wmma.load.a.sync.aligned.m16n16k16.row.shared.bf16`.
///
/// # Safety
///
/// * Must be called by all 32 threads of the warp.
/// * `src` must be 16-byte aligned.
/// * `stride` is in number of elements (not bytes).
#[inline(never)]
pub unsafe fn wmma_load_a_m16n16k16_bf16_row(src: *const u8, stride: u32) -> WmmaFragABf16 {
    let _ = (src, stride);
    unreachable!("wmma_load_a_m16n16k16_bf16_row called outside CUDA kernel context")
}

/// WMMA load of the B operand. m16n16k16, bf16 elements, column-major tile.
/// Lowers to PTX `wmma.load.b.sync.aligned.m16n16k16.col.shared.bf16`.
///
/// # Safety
///
/// * Must be called by all 32 threads of the warp.
/// * `src` must be 16-byte aligned.
/// * `stride` is in number of elements (not bytes).
#[inline(never)]
pub unsafe fn wmma_load_b_m16n16k16_bf16_col(src: *const u8, stride: u32) -> WmmaFragBBf16 {
    let _ = (src, stride);
    unreachable!("wmma_load_b_m16n16k16_bf16_col called outside CUDA kernel context")
}

/// Raw WMMA MMA `D = A * B + C` for m16n16k16 bf16xbf16 -> f32 with
/// individual register-valued lane arguments. Prefer
/// [`wmma_mma_m16n16k16_bf16_bf16_f32`] in user code.
/// Lowers to PTX `wmma.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32`.
///
/// # Safety
///
/// Must be called by all 32 threads of the warp.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub unsafe fn wmma_mma_m16n16k16_bf16_bf16_f32_raw(
    a0: u32, a1: u32, a2: u32, a3: u32,
    b0: u32, b1: u32, b2: u32, b3: u32,
    c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32,
) -> WmmaAccF32 {
    let _ = (a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7);
    unreachable!("wmma_mma_m16n16k16_bf16_bf16_f32_raw called outside CUDA kernel context")
}

/// WMMA MMA `D = A * B + C` for m16n16k16 bf16xbf16 -> f32.
///
/// # Safety
///
/// Must be called by all 32 threads of the warp.
#[inline(always)]
pub unsafe fn wmma_mma_m16n16k16_bf16_bf16_f32(
    a: WmmaFragABf16,
    b: WmmaFragBBf16,
    c: WmmaAccF32,
) -> WmmaAccF32 {
    unsafe {
        wmma_mma_m16n16k16_bf16_bf16_f32_raw(
            a[0], a[1], a[2], a[3],
            b[0], b[1], b[2], b[3],
            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
        )
    }
}

/// Raw WMMA store of the D accumulator with individual lane arguments.
/// Prefer [`wmma_store_d_m16n16k16_bf16_f32_row`] in user code.
/// Lowers to PTX `wmma.store.d.sync.aligned.m16n16k16.row.f32`.
///
/// # Safety
///
/// * Must be called by all 32 threads of the warp.
/// * `dst` must be 16-byte aligned.
/// * `stride` is in number of elements (not bytes).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub unsafe fn wmma_store_d_m16n16k16_bf16_f32_row_raw(
    dst: *mut f32,
    stride: u32,
    d0: f32, d1: f32, d2: f32, d3: f32, d4: f32, d5: f32, d6: f32, d7: f32,
) {
    let _ = (dst, stride, d0, d1, d2, d3, d4, d5, d6, d7);
    unreachable!("wmma_store_d_m16n16k16_bf16_f32_row_raw called outside CUDA kernel context")
}

/// WMMA store of the D accumulator. m16n16k16, f32 elements, row-major tile.
///
/// # Safety
///
/// * Must be called by all 32 threads of the warp.
/// * `dst` must be 16-byte aligned.
/// * `stride` is in number of elements (not bytes).
#[inline(always)]
pub unsafe fn wmma_store_d_m16n16k16_bf16_f32_row(
    dst: *mut f32,
    frag: WmmaAccF32,
    stride: u32,
) {
    unsafe {
        wmma_store_d_m16n16k16_bf16_f32_row_raw(
            dst, stride,
            frag[0], frag[1], frag[2], frag[3],
            frag[4], frag[5], frag[6], frag[7],
        )
    }
}
