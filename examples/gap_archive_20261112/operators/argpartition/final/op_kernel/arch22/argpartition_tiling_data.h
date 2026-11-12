/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file argpartition_tiling_data.h
 * \brief TilingData structure shared by the Argpartition host tiling code
 *        (op_host/arch22/argpartition_tiling.cpp) and the arch22 kernel
 *        (op_kernel/argpartition_arch22.cpp plus op_kernel/arch22/argpartition.h).
 *
 * The structure is a plain, trivially-copyable POD.  The host fills every
 * field, serializes the structure through the raw tiling buffer, and the
 * kernel reads it back directly from its GM tiling buffer (the
 * GET_TILING_DATA_WITH_STRUCT macro would stage it through UB address 0,
 * which is where the kernel keeps its row buffers).  Consequently the
 * host and the kernel must include this very header: no field may be added,
 * removed or reordered in one place only.
 *
 * Semantic contract of the bounded, accuracy-first candidate implemented on
 * top of this structure:
 *   - input  : float32, C-contiguous.  The measured admission profile is
 *              rank 2, shape [32, 16], no broadcasting.
 *   - output : int64, C-contiguous, same shape as the input.
 *   - kth    : static literal 3, expressed here as a zero-based position
 *              (kth = 3) with 0 <= kth < axisLen.
 *   - axis   : static literal 1, normalized to 1 for a rank 2 input.
 *   - result : for every slice along `axis`, the emitted index tensor is a
 *              permutation of the slice positions that satisfies the NumPy
 *              argpartition partition property: positions strictly before
 *              `kth` hold elements no greater than the kth element, the kth
 *              position holds the element that would be at that position after
 *              a full sort, and positions strictly after `kth` hold elements
 *              no smaller than the kth element.
 *
 * Anything outside that bounded scope (other dtypes, other ranks, other
 * shapes, other axes, non-contiguous layouts, out= / other keyword arguments)
 * is rejected by the host API layer.  Such inputs are never silently cast to
 * float32, never silently redirected to a host CPU implementation, and never
 * claimed as qualified before device-side numerical validation.
 *
 * Slices processed by the kernel are the (outer, inner) pairs of the tensor:
 *   slice index s = o * innerNum + i,  0 <= s < sliceNum = outerNum * innerNum
 * where `o` enumerates dims before `axis` and `i` enumerates dims after
 * `axis`.  One slice therefore owns axisLen contiguous float32 values located
 * at input offset (o * axisLen * innerNum + i) and axisLen contiguous int64
 * values located at output offset (o * axisLen * innerNum + i).
 */

#ifndef ARGPARTITION_TILING_DATA_H
#define ARGPARTITION_TILING_DATA_H

#include <cstdint>

// Upper bound on the tensor rank understood by the shared tiling contract.
// The current candidate only admits rank 2, but the bound is part of the
// host/kernel agreement so the host can reject an out-of-scope rank before a
// kernel launch instead of relying on an implicit assumption.
constexpr int32_t ARGPARTITION_TILING_MAX_DIMS = 8;

// Upper bound on one axis slice: 32 float32 elements (128 bytes).  The arch22
// scalar selection path keeps one float32 row and one int64 index row in UB,
// so the host rejects axisLen > ARGPARTITION_TILING_SORT_BLOCK before launch.
// The scalar path reproduces NumPy's introselect permutation exactly; it does
// not pad the row with +inf.
constexpr int32_t ARGPARTITION_TILING_SORT_BLOCK = 32;

// 32-byte alignment expressed in float32 elements (32 / sizeof(float) = 8).
constexpr int64_t ARGPARTITION_TILING_FP32_ALIGN = 8;

// 32-byte alignment expressed in int64 elements (32 / sizeof(int64_t) = 4).
constexpr int64_t ARGPARTITION_TILING_I64_ALIGN = 4;

struct ArgpartitionTilingData {
    // ---------------------------------------------------------------------
    // Problem geometry.  Filled from the runtime input shape by the host
    // tiling function, consumed by the kernel to derive GM offsets and loop
    // bounds.
    // ---------------------------------------------------------------------
    int64_t totalNum = 0;    // number of elements in the input tensor
    int64_t outerNum = 0;    // product of dims before `axis`
    int64_t axisLen = 0;     // dims[axis]: candidates per independent slice
    int64_t innerNum = 0;    // product of dims after `axis`
    int64_t sliceNum = 0;    // outerNum * innerNum, number of independent slices
    int64_t kth = 0;         // zero-based normalized kth, 0 <= kth < axisLen

    // ---------------------------------------------------------------------
    // Work split.  Both factors are expressed in slices, not in elements:
    // one slice is one axisLen-long partition problem.
    // ---------------------------------------------------------------------
    int64_t blockFactor = 0;  // slices handled by one AI Core block
    int64_t ubFactor = 0;     // slices handled by one UB pipeline iteration

    // ---------------------------------------------------------------------
    // Alignment helpers.  alignedAxisLen is axisLen rounded up so that both
    // the float32 input buffer and the int64 output buffer start on a 32-byte
    // boundary in UB; the kernel fills the gap explicitly before Sort.
    // ---------------------------------------------------------------------
    int64_t alignedAxisLen = 0;  // axisLen rounded up to FP32/I64 alignment

    // ---------------------------------------------------------------------
    // Shape metadata used for host-side rejection checks and diagnostics.
    // ---------------------------------------------------------------------
    int32_t axis = 0;          // normalized axis, 0 <= axis < rank
    int32_t rank = 0;          // tensor rank, rank <= ARGPARTITION_TILING_MAX_DIMS
    int32_t sortBlockLen = 0;  // runtime copy of ARGPARTITION_TILING_SORT_BLOCK
    int32_t reserved0 = 0;     // keeps the two 32-bit words 8-byte aligned

    // Original logical shape, zero-padded beyond `rank`.  Kept so the kernel
    // and any post-launch diagnostic observe the same shape the host admitted.
    int64_t dims[ARGPARTITION_TILING_MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
};

#endif // ARGPARTITION_TILING_DATA_H