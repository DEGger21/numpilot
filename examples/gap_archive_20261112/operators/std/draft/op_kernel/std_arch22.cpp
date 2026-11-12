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
 * \file std_arch22.cpp
 * \brief Std operator kernel entry for Ascend910B3 (arch22).
 *
 * Public API -> implementation mapping:
 *   asnumpy.std
 *     -> Python wrapper in asnumpy
 *     -> pybind symbol for std
 *     -> aclnnStdGetWorkspaceSize / aclnnStd
 *     -> l0op::Std
 *     -> std_arch22 kernel entry (this file)
 *     -> NsStd::Std<D_T_X>
 *
 * Admitted profile for this unverified draft:
 *   x: float32, rank 2, C-contiguous, shape [2048, 2048]
 *   axis: 1 (the observed call site; host rejects other axes)
 *   y: float32, rank 1, shape [2048], single tensor return
 *
 * The host definition, infer-shape, tiling, ACLNN layer, pybind layer and
 * NumPy golden tests must reject every other dtype, rank, layout, axis and
 * unsupported keyword before this kernel is launched.  This file intentionally
 * contains no fallback path and no implicit dtype conversion.
 *
 * The kernel entry symbol is named std_arch22 rather than std because the
 * latter would collide with the C++ namespace std in the device compilation
 * unit.  The build system resolves the actual __global__ symbol from this
 * source file; the GE OpType remains Std and the public file stem remains std.
 */

#include "arch22/std.h"
#include "arch22/std_tiling_data.h"

template <typename D_T_X>
__global__ __aicore__ void std_arch22(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StdTilingData);
    GET_TILING_DATA_WITH_STRUCT(StdTilingData, tilingData, tiling);

    NsStd::Std<D_T_X> op;
    op.Init(x, y, &tilingData);
    op.Process();
}