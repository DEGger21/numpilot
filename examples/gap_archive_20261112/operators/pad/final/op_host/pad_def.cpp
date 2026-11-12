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
 * \file pad_def.cpp
 * \brief Pad operator definition (registry-invoke draft, unverified).
 *
 * Semantic target: numpy.pad(x, pad_width, mode='constant')  ->  asnumpy.pad
 *
 * Declared, intentionally bounded surface of this draft candidate:
 *   Input  "x"        : float32 only. Other data dtypes are NOT silently cast; the
 *                       ACLNN layer must reject them explicitly and no CPU/NumPy
 *                       fallback is permitted.
 *   Attr   "paddings" : flattened int64 list {dim0_before, dim0_after, dim1_before,
 *                       dim1_after}. The L0 converts the aclnn aclIntArray into this
 *                       attribute; no index tensor crosses the device boundary because
 *                       the pads only participate in host-side shape and tiling.
 *   Attr   "mode"     : string, always "constant" from the L0. Only the "constant"
 *                       mode is in scope for this candidate (constant fill value 0,
 *                       matching the NumPy default constant_values=0). "reflect"/"edge"/
 *                       "wrap"/"symmetric"/"linear_ramp"/... are rejected by op_api.
 *   Output "y"        : float32, dtype follows "x", shape = x.shape + paddings.
 *
 * Both dtype lists below are aligned index-by-index, so the registry generates exactly
 * two supported combinations:
 *   index 0: x=DT_FLOAT, paddings=DT_INT32, y=DT_FLOAT
 *   index 1: x=DT_FLOAT, paddings=DT_INT64, y=DT_FLOAT
 *
 * Shape rules stay runtime_profile_required: this file declares no fixed rank/shape,
 * and qualification is limited to the measured profiles (float32, rank 2, C contiguous).
 */

#include "register/op_def_registry.h"

namespace ops {
class Pad : public OpDef {
public:
    explicit Pad(const char* name) : OpDef(name)
    {
        this->Input("x")                                        // padded data tensor
            .ParamType(REQUIRED)                                // required input
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})             // float32 only (no implicit cast)
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})             // ND layout
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) // ND layout for unknown shape
            .AutoContiguous();                                  // contiguous input is guaranteed

        this->Output("y")                                       // padded result
            .ParamType(REQUIRED)                                // required output
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})             // mirrors x, never a comparison dtype
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})             // ND layout
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) // ND layout for unknown shape
            .AutoContiguous();                                  // contiguous output is guaranteed

        this->Attr("paddings")                                  // flattened (before, after) per axis
            .AttrType(REQUIRED)                                 // shape and tiling depend on these values
            .ListInt({});                                       // attr index 0, int64 list

        this->Attr("mode")                                      // numpy.pad keyword argument
            .AttrType(REQUIRED)                                 // always supplied by the L0 launch
            .String("constant");                                // only constant reaches the kernel

        // Target SoC for this draft: Ascend910B3 (covered by the "ascend910b" config).
        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)   // compile-time static tiling for this binary
            .DynamicFormatFlag(false)                     // ND only, no dynamic format propagation
            .DynamicRankSupportFlag(false)                // rank is known: pad offsets are rank-indexed
            .DynamicShapeSupportFlag(true)                // dims may be dynamic, rank is fixed
            .NeedCheckSupportFlag(false)                  // no extra support check pass
            .PrecisionReduceFlag(true);                   // no low-precision fast path
        // Kernel entry is op_kernel/pad.cpp -> __global__ __aicore__ void pad(...).
        aicoreConfig910B.ExtendCfgInfo("opFile.value", "pad");
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);
    }
};

OP_ADD(Pad); // register the operator definition into the operator information library
} // namespace ops