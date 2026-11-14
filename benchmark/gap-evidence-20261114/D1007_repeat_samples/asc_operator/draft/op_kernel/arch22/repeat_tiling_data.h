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
 * \file repeat_tiling_data.h
 * \brief TilingData definition shared by Repeat host tiling and the arch22 kernel.
 *
 * The kernel maps an output element to its input element using the normalized
 * decomposed shape:
 *   output linear index = ((outer * (axisSize * repeats) + axisOut) * innerSize) + inner
 *   input linear index  = ((outer * axisSize + axisOut / repeats) * innerSize) + inner
 *
 * Only scalar repeats and C-contiguous float32 tensors are admitted by the
 * current unverified candidate scope.
 */

#ifndef REPEAT_TILING_DATA_H
#define REPEAT_TILING_DATA_H

#include <cstdint>

struct RepeatTilingData {
    // Total number of elements in the output tensor.
    int64_t totalNum = 0;

    // Number of output elements assigned to each vector core.
    int64_t blockFactor = 0;

    // Number of output elements processed in each UB iteration.
    int64_t ubFactor = 0;

    // Product of all input dimensions before the normalized axis.
    int64_t outerSize = 0;

    // Input extent along the normalized axis.
    int64_t axisSize = 0;

    // Product of all input dimensions after the normalized axis.
    int64_t innerSize = 0;

    // Scalar repeat count. The observed admitted profile uses 3.
    int64_t repeats = 0;
};

#endif // REPEAT_TILING_DATA_H