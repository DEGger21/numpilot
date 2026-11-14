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
 * \file argmax_tiling_key.h
 * \brief Argmax tiling template parameter declaration for arch22.
 *
 * Supported template arguments:
 *   - D_T_X: input datatype; this candidate admits float32 only.
 *   - BUFFER_MODE: 0 = single buffer, 1 = double buffer.
 *
 * The kernel entry template parameters must match this declaration order:
 *   template <typename D_T_X, int BUFFER_MODE>
 */

#ifndef __ARGMAX_TILING_KEY_H__
#define __ARGMAX_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define ARGMAX_BUFFER_MODE_SINGLE 0
#define ARGMAX_BUFFER_MODE_DOUBLE 1

ASCENDC_TPL_ARGS_DECL(Argmax,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST,
                          ARGMAX_BUFFER_MODE_SINGLE, ARGMAX_BUFFER_MODE_DOUBLE)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST,
                             ARGMAX_BUFFER_MODE_SINGLE, ARGMAX_BUFFER_MODE_DOUBLE)
    ),
);

#endif // __ARGMAX_TILING_KEY_H__