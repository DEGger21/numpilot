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
 * \file ediff1d_def.cpp
 * \brief Ediff1d operator definition.
 *
 * Public NumPy signature:
 *   numpy.ediff1d(a)
 * Canonical mapping:
 *   Python argument a -> Ascend input 0 ("x")
 *   Returned tensor   -> Ascend output 0 ("y")
 *
 * This candidate is intentionally restricted to the observed float32
 * runtime profile. Other dtypes are rejected by the operator definition
 * and must not be silently cast.
 */

#include "register/op_def_registry.h"

namespace ops {
class Ediff1d : public OpDef {
public:
    explicit Ediff1d(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);

        this->AICore().AddConfig("ascend910b", aicoreConfig910B);
    }
};

OP_ADD(Ediff1d);
} // namespace ops