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
 * \file smooth_l1_loss_tiling_key.h
 * \brief SmoothL1Loss 算子 TilingKey 模板参数定义（arch22 / ascend910b）
 *
 * 模板参数（DECL 定义 / SEL 选择，与 kernel 入口模板参数一一对应）：
 *   - D_T_X: 输入数据类型，仅 float32（与 def 文件 DataTypeList 一致）
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 *
 * 宿主侧 tiling 通过 GET_TPL_TILING_KEY / ASCENDC_TPL_SEL_PARAM 选择模板实参，
 * kernel 入口模板为：
 *   template <typename D_T_X, int BUFFER_MODE>
 *   __global__ __aicore__ void smooth_l1_loss(...)
 */

#ifndef __SMOOTH_L1_LOSS_TILING_KEY_H__
#define __SMOOTH_L1_LOSS_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// 缓冲模式取值
#define SMOOTH_L1_LOSS_BUFFER_MODE_SINGLE 0  // 单缓冲
#define SMOOTH_L1_LOSS_BUFFER_MODE_DOUBLE 1  // 双缓冲

ASCENDC_TPL_ARGS_DECL(SmoothL1Loss,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST,
                          SMOOTH_L1_LOSS_BUFFER_MODE_SINGLE,
                          SMOOTH_L1_LOSS_BUFFER_MODE_DOUBLE)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST,
                             SMOOTH_L1_LOSS_BUFFER_MODE_SINGLE,
                             SMOOTH_L1_LOSS_BUFFER_MODE_DOUBLE)
    ),
);

#endif // __SMOOTH_L1_LOSS_TILING_KEY_H__