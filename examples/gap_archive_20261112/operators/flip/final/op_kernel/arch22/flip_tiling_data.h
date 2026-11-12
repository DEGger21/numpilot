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
 * \file flip_tiling_data.h
 * \brief Shared tiling-data structure for the arch22 Flip candidate.
 *
 * Bounded support scope for this candidate:
 *   - float32 input tensors
 *   - rank-2 C-contiguous tensors
 *   - axis normalized to dim 0, dim 1, both dims, or empty
 *
 * The kernel processes output rows. Each AI core owns at most
 * `blockFactor` consecutive output rows. Within a row, at most
 * `ubFactor` float32 elements are moved through UB per tile.
 *
 * `flipDim0` and `flipDim1` are the authoritative effective flip mask.
 * `axis` is retained for diagnostics and host/kernel agreement; kernels
 * should use the flip mask for address calculation.
 */

#ifndef FLIP_TILING_DATA_H
#define FLIP_TILING_DATA_H

#include <cstdint>

struct FlipTilingData {
    // Total logical elements: rows * cols. Zero for an empty tensor.
    int64_t totalNum = 0;

    // Logical rank-2 shape. `cols` may be zero only when totalNum is zero.
    int64_t rows = 0;
    int64_t cols = 0;

    // Number of output rows assigned to one AI core. The final core may own
    // fewer rows. The host derives this from the platform core count.
    int64_t blockFactor = 0;

    // Maximum number of float32 elements copied into UB per tile. This is
    // always >= 1 for non-empty tensors and <= cols.
    int64_t ubFactor = 0;

    // Effective flip mask: 1 means the corresponding dimension is reversed.
    int32_t flipDim0 = 0;
    int32_t flipDim1 = 0;

    // Normalized axis encoding for diagnostics:
    //   0  -> dim 0 only
    //   1  -> dim 1 only
    //   2  -> both dims (axis=None or a two-axis tuple)
    //  -1  -> no dims (axis=())
    // Negative single axes are normalized by the host before this value is set.
    int32_t axis = 0;

    // Explicit padding, always zero. Keeps the initial field layout stable.
    int32_t reserved = 0;
};

#endif  // FLIP_TILING_DATA_H