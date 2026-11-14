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
 * \file repeat_tiling_key.h
 * \brief Tiling template arguments for Repeat (arch22).
 *
 * This header is the single authoritative declaration of the template
 * parameters used by both the host tiling selection and the kernel entry.
 *
 * The kernel entry must be declared with exactly these template
 * parameters and in this order:
 *     template <typename D_T_X, int BUFFER_MODE>
 *
 * Current bounded support:
 *   - input/output dtype: float32 only (D_T_X = C_DT_FLOAT)
 *   - buffer mode: single buffer only (BUFFER_MODE = 0)
 *
 * Other input dtypes are outside the admitted scope and must be rejected
 * by the operator definition and host tiling, not silently cast.
 */

#ifndef __REPEAT_TILING_KEY_H__
#define __REPEAT_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Repeat,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)
    )
);

#endif // __REPEAT_TILING_KEY_H__