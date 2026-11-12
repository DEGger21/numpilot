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
 * \file ediff1d_arch22.cpp
 * \brief Ediff1d 算子 kernel 入口（arch22 架构）
 *
 * 模板参数说明（与 ediff1d_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T_X: 输入数据类型；当前已确认 profile 为 float32
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 *
 * 语义：按 C 顺序展平输入，输出 y[i] = x[i+1] - x[i]。
 * 输入/输出均为单张量，参数顺序固定为 inputs -> outputs -> workspace -> tiling。
 */

#include "arch22/ediff1d.h"

extern "C" __global__ __aicore__ void ediff1d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(Ediff1dTilingData);
    GET_TILING_DATA_WITH_STRUCT(Ediff1dTilingData, tilingData, tiling);
    // float32 and single buffering are the only admitted instantiation.
    NsEdiff1d::Ediff1d<float, 0> op;
    op.Init(x, y, &tilingData);
    op.Process();
}