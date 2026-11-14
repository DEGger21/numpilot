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
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t outCols = 0;
    int32_t axis = 0;
    int32_t reserved = 0;
};

#endif /* DIFF_TILING_DATA_H */