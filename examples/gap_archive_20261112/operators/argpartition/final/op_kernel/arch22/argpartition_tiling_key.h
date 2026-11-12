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
 * \file argpartition_tiling_key.h
 * \brief Tiling template parameter declaration for the arch22 Argpartition kernel.
 *
 * The current candidate is intentionally specialized to the only measured input
 * profile: float32 data, rank 2, shape [32, 16], C-contiguous, with kth=3 and
 * axis=1. Host-side validation rejects every other input dtype and unsupported
 * axis/shape/signature before tiling is produced, so no additional algorithm or
 * buffer template parameter is exposed here.
 *
 * The single declared template axis is the input datatype:
 *   - D_T_X maps to the datatype of input 0, currently C_DT_FLOAT only.
 *
 * The kernel entry and host ASCENDC_TPL_SEL_PARAM usage must both use the
 * D_T_X template parameter declared below, in this order.
 */

#ifndef __ARGPARTITION_TILING_KEY_H__
#define __ARGPARTITION_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(Argpartition,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, ASCENDC_TPL_INPUT(0))
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)
    )
);

#endif /* __ARGPARTITION_TILING_KEY_H__ */