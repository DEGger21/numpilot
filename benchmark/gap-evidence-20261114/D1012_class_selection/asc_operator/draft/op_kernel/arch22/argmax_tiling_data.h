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
 * \file argmax_tiling_data.h
 * \brief Authoritative TilingData schema for the Argmax arch22 kernel.
 *
 * The host tiling implementation and the kernel must both include this
 * header. Host tiling code must assign only the fields declared below.
 *
 * Axis decomposition for a normalized axis in [0, rank):
 *   outerSize  = product(shape[0:axis])
 *   reduceSize = shape[axis]
 *   innerSize  = product(shape[axis + 1:rank])
 *   totalOut   = outerSize * innerSize
 *
 * For an output index o in [0, totalOut):
 *   outerIdx  = o / innerSize
 *   innerIdx  = o % innerSize
 *   inputBase = outerIdx * reduceSize * innerSize + innerIdx
 *   candidate r (0 <= r < reduceSize) is input[inputBase + r * innerSize]
 * The selected index is stored in output[o] as an int64 value.
 *
 * This schema intentionally supports the observed float32 [1024, 1024]
 * axis=1 profile (outerSize=1024, reduceSize=1024, innerSize=1). The host
 * tiling layer is responsible for rejecting profiles outside the declared
 * support scope before launching the kernel.
 */

#ifndef ARGMAX_TILING_DATA_H
#define ARGMAX_TILING_DATA_H

#include <cstdint>

struct ArgmaxTilingData {
    // Normalized reduction-axis decomposition.
    int64_t rank = 0;
    int64_t axis = 0;
    int64_t outerSize = 0;
    int64_t reduceSize = 0;
    int64_t innerSize = 0;

    // Number of output elements: outerSize * innerSize.
    int64_t totalOut = 0;

    // Multi-core and UB tiling factors, measured in output elements.
    int64_t blockFactor = 0;  // output elements assigned to one AI core
    int64_t ubFactor = 0;     // output elements processed per UB iteration
    int64_t usedCoreNum = 0;  // number of AI cores with valid work
};

#endif  // ARGMAX_TILING_DATA_H