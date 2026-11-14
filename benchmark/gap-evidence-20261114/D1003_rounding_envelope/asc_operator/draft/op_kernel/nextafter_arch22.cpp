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
 * \file nextafter_arch22.cpp
 * \brief Nextafter 算子 kernel 入口（arch22 架构）
 *
 * 模板参数说明（必须与 nextafter_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义一致）：
 *   - D_T_X:      输入数据类型；本候选仅支持 float32
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 *
 * 参数顺序固定为：输入1、输入2、输出、workspace、tiling。
 * 本入口只做初始化与调度，计算逻辑位于 arch22/nextafter.h 的 NsNextafter::Nextafter 中。
 */

#include "arch22/nextafter.h"

template <typename D_T_X = float, int BUFFER_MODE = 1>
__global__ __aicore__ void nextafter(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NextafterTilingData);
    GET_TILING_DATA_WITH_STRUCT(NextafterTilingData, tilingData, tiling);
    NsNextafter::Nextafter<D_T_X, BUFFER_MODE> op;
    op.Init(x1, x2, y, &tilingData);
    op.Process();
}