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
 * \file pad_tiling_key.h
 * \brief Tiling template parameter declarations for Pad (arch22).
 *
 * The declared selection is intentionally bounded to the currently admitted
 * profile: float32 input with constant padding mode. Other input dtypes and
 * padding modes must be rejected by the host API/tiling layer rather than
 * silently mapped onto this kernel.
 *
 * Template parameters:
 *   - D_T_X: input tensor data type, bound to input 0.
 *   - PAD_MODE: padding mode selector. Only constant padding is declared here.
 */

#ifndef __PAD_TILING_KEY_H__
#define __PAD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define PAD_TPL_MODE_CONSTANT 0

ASCENDC_TPL_ARGS_DECL(Pad,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(PAD_MODE, 8, ASCENDC_TPL_UI_LIST, PAD_TPL_MODE_CONSTANT)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(PAD_MODE, ASCENDC_TPL_UI_LIST, PAD_TPL_MODE_CONSTANT)
    )
);

#endif