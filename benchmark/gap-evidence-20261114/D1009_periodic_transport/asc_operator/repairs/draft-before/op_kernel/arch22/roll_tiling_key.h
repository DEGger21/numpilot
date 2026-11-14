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
 * \file roll_tiling_key.h
 * \brief Roll 算子 arch22 TilingKey 与模板参数定义
 *
 * 目标 SoC：Ascend910B3
 * 目标 CANN：9.2.0-beta.2
 *
 * 本候选仅声明 float32 输入模板组合。其他输入 dtype 不由本 TilingKey 声明，
 * 必须在 Host 侧 tiling/API 参数校验中显式拒绝，禁止静默转换。
 */

#ifndef __ROLL_TILING_KEY_H__
#define __ROLL_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

/*
 * Kernel 入口模板参数：
 *   D_T_X：输入/输出张量元素类型，来自输入 0 的 dtype。
 *
 * Host tiling 使用 ASCENDC_TPL_SEL_PARAM(context, D_T_X) 选择对应的
 * TilingKey；kernel 入口使用 template <typename D_T_X>。
 */
ASCENDC_TPL_ARGS_DECL(
    Roll,
    ASCENDC_TPL_DATATYPE_DECL(
        D_T_X,
        C_DT_FLOAT,
        ASCENDC_TPL_INPUT(0)
    )
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(
            D_T_X,
            C_DT_FLOAT
        )
    )
);

#endif