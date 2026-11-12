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
 *     -> numpy_std kernel entry (this file)
 *     -> NsStd::Std<float>
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
 * The kernel entry symbol is numpy_std: opbuild derives origin_func_name from
 * the OpType via optype_snake_ex (NumpyStd -> numpy_std) and calls it without
 * template arguments.  The OpType cannot be `Std`: the device translation
 * unit includes libstdc++ headers, so a global function named std collides
 * with the C++ namespace std.  Only the internal GE identity changes; the
 * public ACLNN interface and the NumPy API identity are unchanged.
 */

#include "arch22/std.h"
#include "arch22/std_tiling_data.h"

extern "C" __global__ __aicore__ void numpy_std(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StdTilingData);
    GET_TILING_DATA_WITH_STRUCT(StdTilingData, tilingData, tiling);

    NsStd::Std<float> op;
    op.Init(x, y, &tilingData);
    op.Process();
}