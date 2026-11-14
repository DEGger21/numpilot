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
 * \file nextafter_tiling_data.h
 * \brief Authoritative tiling data schema shared by Nextafter host tiling and kernel.
 *
 * Supported bounded profile for this candidate: two C-contiguous float32 inputs
 * of shape [1024, 1024] and one float32 output of shape [1024, 1024]. The host
 * tiling implementation and kernel entry must both use this single struct
 * definition; no duplicate tiling-data schema is permitted.
 */

#ifndef NEXTAFTER_TILING_DATA_H
#define NEXTAFTER_TILING_DATA_H

#include <cstdint>

struct NextafterTilingData {
    // Total number of elements in the flattened output tensor.
    int64_t totalNum = 0;

    // Number of contiguous elements assigned to each AI core.
    int64_t blockFactor = 0;

    // Number of elements processed per UB tile inside a core.
    int64_t ubFactor = 0;
};

#endif // NEXTAFTER_TILING_DATA_H