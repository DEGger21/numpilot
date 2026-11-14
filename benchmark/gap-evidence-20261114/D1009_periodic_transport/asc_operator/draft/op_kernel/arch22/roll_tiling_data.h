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
 * \file roll_tiling_data.h
 * \brief Shared tiling data for the arch22 Roll operator.
 *
 * Supported admitted profile for this candidate:
 *   - float32 input/output
 *   - rank-2 C-contiguous tensors
 *   - integer axis in the 2-D range [0, 1]
 *
 * `shift` is the host-normalised shift for the selected axis and is stored
 * in the range [0, extent(axis)).  The kernel must reject any invocation
 * whose runtime dtype, rank, layout, or axis falls outside the admitted
 * profile rather than silently casting or falling back to CPU.
 */

#ifndef ROLL_TILING_DATA_H
#define ROLL_TILING_DATA_H

#include <stdint.h>

struct RollTilingData {
    int64_t totalNum = 0;
    int64_t rowLen = 0;
    int64_t shiftOffset = 0;
    int64_t rowCount = 0;
    int64_t blockRows = 0;
    int64_t ubFactor = 0;
};

#endif // ROLL_TILING_DATA_H