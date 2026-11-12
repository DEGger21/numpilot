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
 * \file std_tiling_data.h
 * \brief Host/kernel shared tiling-data layout for asnumpy.std (axis=1 row reduction).
 *
 * The admitted profile is a rank-2, C-contiguous float32 tensor with shape
 * [totalRows, totalCols].  One float32 output row is produced for each input
 * row.  All element counts below are expressed in elements, never in bytes.
 */

#ifndef OP_KERNEL_ARCH22_STD_TILING_DATA_H_
#define OP_KERNEL_ARCH22_STD_TILING_DATA_H_

#include <cstdint>

struct StdTilingData {
    // Number of output rows (the first input dimension).
    int64_t totalRows = 0;

    // Number of reduced columns per row (the second input dimension).
    int64_t totalCols = 0;

    // Number of output rows assigned to each launched AI core.
    // The last core may own fewer rows when totalRows is not divisible.
    int64_t blockFactor = 0;

    // Number of input columns processed by one inner UB iteration.
    // For the admitted [2048, 2048] profile this is at most totalCols.
    int64_t ubFactor = 0;

    // Number of AI cores launched for this invocation.
    int64_t usedCoreNum = 0;
};

#endif // OP_KERNEL_ARCH22_STD_TILING_DATA_H_