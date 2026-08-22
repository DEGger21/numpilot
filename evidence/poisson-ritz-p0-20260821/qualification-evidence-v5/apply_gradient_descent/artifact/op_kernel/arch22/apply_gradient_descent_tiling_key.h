/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_gradient_descent_tiling_key.h
 * \brief ApplyGradientDescent 算子 TilingKey 模板参数定义（arch22 / ascend910b）
 *
 * 模板参数（DECL 定义 / SEL 选择，与 kernel 入口模板参数一一对应）：
 *   - D_T_X: 数据类型，仅 float32（ admitted dtype: float32 ）
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 */

#ifndef __APPLY_GRADIENT_DESCENT_TILING_KEY_H__
#define __APPLY_GRADIENT_DESCENT_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define APPLY_GRADIENT_DESCENT_BUFFER_MODE_SINGLE 0
#define APPLY_GRADIENT_DESCENT_BUFFER_MODE_DOUBLE 1

ASCENDC_TPL_ARGS_DECL(ApplyGradientDescent,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST,
                          APPLY_GRADIENT_DESCENT_BUFFER_MODE_SINGLE,
                          APPLY_GRADIENT_DESCENT_BUFFER_MODE_DOUBLE)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST,
                             APPLY_GRADIENT_DESCENT_BUFFER_MODE_SINGLE,
                             APPLY_GRADIENT_DESCENT_BUFFER_MODE_DOUBLE)
    ),
);

#endif // __APPLY_GRADIENT_DESCENT_TILING_KEY_H__