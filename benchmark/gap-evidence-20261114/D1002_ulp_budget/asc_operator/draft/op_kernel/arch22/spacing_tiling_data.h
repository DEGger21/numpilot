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
 * \file spacing_tiling_data.h
 * \brief Tiling data structure shared by the Spacing host tiling and arch22 kernel.
 */

#ifndef SPACING_TILING_DATA_H
#define SPACING_TILING_DATA_H

#include <cstdint>

struct SpacingTilingData {
    int64_t totalNum = 0;     // Total number of float32 elements in the input/output tensor.
    int64_t blockFactor = 0;  // Number of elements assigned to each AI Core.
    int64_t ubFactor = 0;     // Number of elements processed per UB loop iteration.
};

#endif // SPACING_TILING_DATA_H