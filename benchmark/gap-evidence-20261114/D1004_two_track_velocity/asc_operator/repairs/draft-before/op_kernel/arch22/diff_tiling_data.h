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
 * \file diff_tiling_data.h
 * \brief Tiling data shared by Diff host tiling and the arch22 kernel.
 *
 * Supported semantic slice for this candidate:
 *   - float32, C-contiguous input;
 *   - one positional tensor argument and the keyword argument ``axis``;
 *   - n == 1, matching numpy.diff(a, axis=axis);
 *   - normalized axis in [0, rank).
 *
 * The host tiling code splits the input shape into three logical extents
 * around the normalized axis:
 *
 *   input  shape: (outerSize, axisSize, innerSize)
 *   output shape: (outerSize, outputAxisSize, innerSize)
 *
 * where outputAxisSize == axisSize - 1.  For an output linear index
 * constructed as (outer, axisOut, inner), the kernel reads:
 *
 *   lhs = input[((outer * axisSize) + axisOut) * innerSize + inner]
 *   rhs = input[((outer * axisSize) + axisOut + 1) * innerSize + inner]
 *   out[...] = rhs - lhs
 *
 * Host tiling MUST populate every field below and satisfy:
 *   outputAxisSize == axisSize - 1
 *   totalNum == outerSize * outputAxisSize * innerSize
 *   blockFactor == ceil(totalNum / usedCoreNum)
 *   ubFactor > 0 when totalNum > 0
 *
 * totalNum counts output elements, not input elements.  When totalNum is
 * zero (for example an empty tensor or axisSize == 1), blockFactor and
 * ubFactor may be zero; the kernel exits without accessing global memory.
 */

#ifndef DIFF_TILING_DATA_H
#define DIFF_TILING_DATA_H

#include <stdint.h>

struct DiffTilingData {
    /* Number of output elements produced by numpy.diff along the selected axis. */
    int64_t totalNum = 0;

    /* Maximum number of output elements assigned to one AI core. */
    int64_t blockFactor = 0;

    /* Number of output elements handled by one CopyIn/Compute/CopyOut iteration. */
    int64_t ubFactor = 0;

    /* Product of input dimensions preceding the normalized axis. */
    int64_t outerSize = 0;

    /* Input extent along the normalized axis. */
    int64_t axisSize = 0;

    /* Product of input dimensions following the normalized axis. */
    int64_t innerSize = 0;

    /* Output extent along the normalized axis; always axisSize - 1. */
    int64_t outputAxisSize = 0;

    /* Normalized non-negative axis in [0, rank). */
    int32_t axis = 0;

    /* Reserved for ABI growth; host tiling sets this to zero. */
    int32_t reserved = 0;
};

#endif /* DIFF_TILING_DATA_H */