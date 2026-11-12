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
 * \file pad_arch22.cpp
 * \brief Pad 算子 kernel 入口（arch22 / Ascend910B3）
 *
 * Repair note (v8): the entry no longer uses GET_TILING_DATA_WITH_STRUCT.
 * That macro stages the tiling struct through UB address 0, which overlaps the
 * first TPipe queue allocation and caused padded rows to read stale tiling
 * bytes.  The kernel now reads its six scalar tiling fields straight from the
 * tiling GM buffer.
 */

#include "arch22/pad.h"

extern "C" __global__ __aicore__ void pad(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

    const __gm__ PadTilingData* tilingData =
        reinterpret_cast<const __gm__ PadTilingData*>(tiling);

    NsPad::Pad<float> op;
    op.Init(x, y, tilingData);
    op.Process();
}
