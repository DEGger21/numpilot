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
 * \file pad_tiling_data.h
 * \brief Tiling data shared by the Pad host tiling implementation and the arch22 kernel.
 *
 * Bounded support for this draft:
 *   - float32 input and output only;
 *   - rank-2, C-contiguous input tensors;
 *   - mode='constant' with the NumPy default constant value 0.0f unless the host
 *     tiling explicitly records a different admitted value.
 *
 * Inputs outside the admitted dtype/shape/signature scope must be rejected by
 * the host API layer before launch. This header intentionally contains no
 * fallback or host-side computation path.
 */

#ifndef PAD_TILING_DATA_H
#define PAD_TILING_DATA_H

#include <stdint.h>

struct PadTilingData {
    // Total number of elements in the logical output tensor.
    int64_t totalNum = 0;

    // Number of output elements assigned to each AI core. The final core may
    // receive fewer elements.
    int64_t blockFactor = 0;

    // Maximum number of elements handled by one UB CopyIn/Compute/CopyOut
    // iteration.
    int64_t ubFactor = 0;

    // Total number of elements in the logical input tensor.
    int64_t totalInputNum = 0;

    // Logical input shape [inputDim0, inputDim1].
    int64_t inputDim0 = 0;
    int64_t inputDim1 = 0;

    // Logical output shape [outputDim0, outputDim1].
    int64_t outputDim0 = 0;
    int64_t outputDim1 = 0;

    // Padding before/after each axis. For an output coordinate (i, j), the
    // corresponding input coordinate is (i - padTop, j - padLeft).
    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;

    // Constant fill value used for mode='constant'. The NumPy default is 0.0f.
    float constantValue = 0.0f;

    // Number of cores actually used by the launch.
    int64_t usedCoreNum = 0;
};

#endif // PAD_TILING_DATA_H