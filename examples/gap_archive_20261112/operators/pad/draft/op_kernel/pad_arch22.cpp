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
 * Supported scope for this draft:
 *   - float32 input and output tensors
 *   - rank-2 C-contiguous tensors
 *   - mode="constant" only
 *
 * The kernel class and tiling-data definition are provided by the paired
 * arch22/pad.h and arch22/pad_tiling_data.h project files. This translation
 * unit contains only the fixed kernel entry required by the registry-invoke
 * template.
 */

#include "arch22/pad.h"

__global__ __aicore__ void pad(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PadTilingData);
    GET_TILING_DATA_WITH_STRUCT(PadTilingData, tilingData, tiling);

    (void)workspace;

    NsPad::Pad op;
    op.Init(x, y, &tilingData);
    op.Process();
}