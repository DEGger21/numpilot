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
 * \file tril_arch22.cpp
 * \brief Tril kernel entry for Ascend910B3 (arch22).
 *
 * NumPy semantic contract:
 *   y[i, j] = x[i, j] for j <= i
 *   y[i, j] = 0      for j > i
 *
 * The public signature has exactly one positional argument and no keyword
 * arguments, so the diagonal offset is fixed to k = 0.  The admitted input
 * profile is a two-dimensional, C-contiguous float32 tensor.  Rejection of
 * other dtypes, ranks, layouts, and unsupported keyword arguments is handled
 * by the host/ACLNN layer before this kernel is launched.
 */

#include "arch22/tril.h"

template <typename D_T_X>
__global__ __aicore__ void tril(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

    REGISTER_TILING_DEFAULT(TrilTilingData);
    GET_TILING_DATA_WITH_STRUCT(TrilTilingData, tilingData, tiling);

    NsTril::Tril<D_T_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}