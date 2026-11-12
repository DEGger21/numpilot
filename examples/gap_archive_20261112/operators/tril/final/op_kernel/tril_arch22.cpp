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
 * \file tril_arch22.cpp
 * \brief Tril kernel entry for Ascend910B3 (arch22).
 *
 * The tiling struct is read directly from its GM buffer, so the entry does
 * not use GET_TILING_DATA_WITH_STRUCT (that macro stages the struct through UB
 * address 0, which is also the kernel's staging buffer base).
 */

#include "arch22/tril.h"

extern "C" __global__ __aicore__ void tril(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    const __gm__ TrilTilingData* tilingData = reinterpret_cast<const __gm__ TrilTilingData*>(tiling);
    NsTril::Tril<float> op;
    op.Init(x, y, tilingData);
    op.Process();
}
