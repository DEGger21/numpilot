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
    // Logical input shape [inH, inW].
    int64_t inH = 0;
    int64_t inW = 0;

    // Logical output shape [outH, outW]:
    //   outH = inH + padTop + padBottom, outW = inW + padLeft + padRight.
    int64_t outH = 0;
    int64_t outW = 0;

    // Padding offsets. For an output coordinate (i, j), the corresponding input
    // coordinate is (i - padTop, j - padLeft).
    int64_t padTop = 0;
    int64_t padLeft = 0;

    // Constant fill value used for mode='constant'. The NumPy default is 0.0f.
    float constantValue = 0.0f;

    // Output rows assigned to each core; the host records it for diagnostics.
    int64_t rowsPerCore = 0;
};

#endif // PAD_TILING_DATA_H