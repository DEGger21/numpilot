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
    /* Total number of elements in the input/output tensor. */
    union {
        int64_t totalNum;
        int64_t totalElements;
        int64_t elementCount;
        int64_t numElements;
    };

    /* Normalised roll amount for the selected axis. */
    union {
        int64_t shift;
        int64_t shiftValue;
        int64_t shiftAmount;
        int64_t rollShift;
    };

    /* Normalised axis index. */
    union {
        int64_t axis;
        int64_t axisValue;
        int64_t axisDim;
        int64_t rollAxis;
    };

    /* Row extent for rank-2 tensors. */
    union {
        int64_t rowCount;
        int64_t rowNum;
        int64_t numRows;
        int64_t rows;
        int64_t dim0;
    };

    /* Column extent for rank-2 tensors. */
    union {
        int64_t colCount;
        int64_t colNum;
        int64_t numCols;
        int64_t cols;
        int64_t dim1;
    };

    /* Number of flat elements assigned to one AI Core. */
    union {
        int64_t blockFactor;
        int64_t blockElements;
        int64_t perCoreElements;
    };

    /* Number of flat elements processed in one UB tile iteration. */
    union {
        int64_t ubFactor;
        int64_t ubElements;
        int64_t tileElements;
        int64_t ubLength;
    };

    /* Runtime tensor rank, retained for explicit validation by the kernel. */
    union {
        int64_t rank;
        int64_t rankValue;
        int64_t dimNum;
    };

    /* Column extent rounded up to the kernel's chosen alignment. */
    union {
        int64_t alignedColCount;
        int64_t alignedColNum;
        int64_t alignedCols;
        int64_t colAlign;
    };

    /* Row extent rounded up to the kernel's chosen alignment. */
    union {
        int64_t alignedRowCount;
        int64_t alignedRowNum;
        int64_t alignedRows;
        int64_t rowAlign;
    };
};

#endif // ROLL_TILING_DATA_H