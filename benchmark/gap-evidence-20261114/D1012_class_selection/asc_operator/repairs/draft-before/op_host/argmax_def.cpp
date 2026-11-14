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
 * \file argmax_def.cpp
 * \brief Argmax operator definition.
 *
 * The bounded support scope for this candidate is:
 *   - input  x: float32, C-contiguous
 *   - output y: int64
 *   - attribute axis: required int64, interpreted according to NumPy axis normalization
 *
 * Unsupported input dtypes and unsupported signatures must be rejected explicitly by
 * the Python and ACLNN entry points; this definition intentionally declares only the
 * observed float32 -> int64 combination.
 */

#include "register/op_def_registry.h"

namespace ops {
class Argmax : public OpDef {
public:
    explicit Argmax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("axis")
            .AttrType(REQUIRED)
            .Int();

        this->ExtendCfgInfo("opFile.value", "argmax");

        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);
    }
};

OP_ADD(Argmax);
} // namespace ops