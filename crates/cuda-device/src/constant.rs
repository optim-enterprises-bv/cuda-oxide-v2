/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant-memory support for CUDA kernels.
//!
//! [`Constant<T>`] is a wrapper for module-scope statics that live in CUDA
//! constant memory (PTX `.const`, address space 4). The host populates the
//! storage via `cuMemcpyHtoD`; device code reads it as if it were an
//! ordinary `static T`, returning a value-typed copy.
//!
//! # Usage
//!
//! Declare the static inside a `#[cuda_module]` and tag it with `#[constant]`:
//!
//! ```ignore
//! use cuda_device::{constant, cuda_module, kernel, thread, Constant, DisjointSlice};
//!
//! #[cuda_module]
//! mod kernels {
//!     use super::*;
//!
//!     #[constant]
//!     static COEFFS: Constant<[f32; 4]> = Constant::UNINIT;
//!
//!     #[kernel]
//!     pub fn apply(mut out: DisjointSlice<f32>) {
//!         let c = COEFFS.get();        // safe read; returns [f32; 4]
//!         let i = thread::index_1d().get();
//!         if let Some(e) = out.get_mut(thread::index_1d()) {
//!             *e = c[0] + c[1] * (i as f32);
//!         }
//!     }
//! }
//! ```
//!
//! Host code overrides the initializer between launches with the
//! macro-generated `set_<NAME>` methods on the loaded module:
//!
//! ```ignore
//! module.set_COEFFS(&stream, &[10.0, 20.0, 30.0, 40.0])?;
//! ```
//!
//! # Why a wrapper type instead of a bare `static`
//!
//! A plain `static COEFFS: [f32; 4] = [1.0; 4];` would be constant-folded by
//! rustc — every read in device code is replaced with the literal initializer
//! values, and the host's `cuMemcpyHtoD` update becomes invisible. Wrapping
//! the storage in [`UnsafeCell`] prevents the fold by signalling interior
//! mutability, restoring the read-from-memory semantics that CUDA constant
//! memory requires.
//!
//! # Soundness: `Sync`
//!
//! Unlike [`SharedArray`](crate::SharedArray) (which is `!Sync` because
//! shared memory is per-block and requires barriers), `Constant<T>` is
//! `Sync`. CUDA constant memory has a single, host-controlled value visible
//! identically to every thread on the device, with no in-kernel writes; a
//! `&Constant<T>` from any thread is sound to read concurrently.

use core::cell::UnsafeCell;

/// A `static`-friendly wrapper that places `T` in CUDA constant memory
/// (`addrspace(4)`).
///
/// See the [module docs](self) for the full usage pattern.
#[repr(transparent)]
pub struct Constant<T: Copy>(UnsafeCell<T>);

// SAFETY: Constant<T> is a host-populated, device-readonly cell. The host
// performs writes via `cuMemcpyHtoD` (synchronized with the calling thread);
// device code only reads. No concurrent writer exists on either side, so
// shared `&Constant<T>` across threads is sound.
unsafe impl<T: Copy + Send> Sync for Constant<T> {}

impl<T: Copy> Constant<T> {
    /// Placeholder value for a `#[constant]` static. The host must call
    /// the macro-generated `set_<NAME>` before any kernel reads the
    /// constant; honoring arbitrary non-zero initializers in codegen is
    /// not yet implemented.
    ///
    /// Mirrors the convention used by [`SharedArray::UNINIT`](crate::SharedArray)
    /// and [`Barrier::UNINIT`](crate::barrier::Barrier): a single placeholder
    /// constant for a type that's expected to be populated out-of-band.
    ///
    /// # Note
    ///
    /// The underlying bytes are zero. `T` must be valid as an all-zeros
    /// bit pattern — this holds for all integer, floating-point,
    /// raw-pointer, and SIMD types and arrays thereof; it does *not* hold
    /// for `NonZero*`, `&T`, or types containing niches that exclude
    /// zero. Using `UNINIT` with such a type produces a value whose
    /// Rust-level invariants are violated, and reading it before the host
    /// populates it is undefined behavior. The constant is not `unsafe`
    /// to access, matching the ergonomic convention of `SharedArray::UNINIT`.
    pub const UNINIT: Self = Constant(UnsafeCell::new(unsafe {
        core::mem::MaybeUninit::<T>::zeroed().assume_init()
    }));

    /// Read the current value.
    ///
    /// Returns a by-value copy of the storage. Safe because constant memory
    /// is read-only from the device — there is no possibility of observing
    /// a torn write from another thread.
    ///
    /// The `UnsafeCell` interior prevents the compiler from hoisting reads
    /// across `set_<NAME>` boundaries, which means a `.get()` inside a hot
    /// loop will re-read on every iteration. For large `T` (e.g.
    /// `Constant<[f32; 1024]>`) call `.get()` once before the loop and
    /// reuse the local:
    ///
    /// ```ignore
    /// let coeffs = COEFFS.get();    // single load, hoisted by you
    /// for i in 0..n { use(coeffs[i % 4]) }
    /// ```
    #[inline(always)]
    pub fn get(&self) -> T {
        // SAFETY: read-only from device, and `T: Copy` means we never alias
        // a mutable reference. The host updates this storage only between
        // kernel launches via `cuMemcpyHtoD`, which is synchronized
        // out-of-band relative to device execution.
        unsafe { *self.0.get() }
    }
}
