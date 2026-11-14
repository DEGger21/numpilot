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
 * \file diff_tiling_key.h
 * \brief Tiling template parameter declaration for the Diff operator on arch22.
 *
 * Kernel entry contract:
 *   template <typename D_T_X>
 *   __global__ __aicore__ void diff(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
 *
 * Only float32 input/output is admitted by this candidate stage. The diff axis
 * is carried in DiffTilingData and is validated by host tiling; it is not a
 * TilingKey template parameter for this accuracy-first implementation.
 */

#ifndef __DIFF_TILING_KEY_H__
#define __DIFF_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Diff,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0))
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)
    )
);

#endif