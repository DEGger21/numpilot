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
 * \file ediff1d_tiling_data.h
 * \brief Ediff1d tiling data shared by the host tiling implementation and the arch22 kernel.
 *
 * NumPy semantics implemented by this operator:
 *   out[i] = in[i + 1] - in[i], for i in [0, inputNum - 1)
 * where the input is treated as a flat, C-contiguous sequence.
 *
 * The tiling fields below are expressed in output elements.  Therefore:
 *   outputNum = totalNum
 *   inputNum  = totalNum + 1
 * and a tile of output elements [offset, offset + count) reads input elements
 * [offset, offset + count] (count + 1 input values).
 */

#ifndef EDIFF1D_TILING_DATA_H
#define EDIFF1D_TILING_DATA_H

#include <cstdint>

struct Ediff1dTilingData {
    // Total number of output differences.  For an input containing N elements this is N - 1.
    int64_t totalNum = 0;

    // Number of output elements assigned to each AI core.  The final core may process fewer.
    int64_t blockFactor = 0;

    // Number of output elements processed per UB loop iteration.  The final iteration may be smaller.
    int64_t ubFactor = 0;
};

#endif // EDIFF1D_TILING_DATA_H