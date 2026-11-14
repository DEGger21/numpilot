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
 * \file spacing_def.cpp
 * \brief Spacing operator definition for NumPy-compatible np.spacing on Ascend C.
 *
 * Supported public signature:
 *   asnumpy.spacing(x)
 *
 * Supported target:
 *   SoC: Ascend910B3 (operator config: ascend910b)
 *   CANN: 9.2.0-beta.2
 *
 * Supported profile for this draft:
 *   input:  float32, rank 2, shape [1024, 1024], C-contiguous
 *   output: float32, same shape and layout as input
 *
 * Other dtypes, ranks, shapes, layouts, broadcasting, and keyword arguments are
 * intentionally outside this draft's admitted scope and must be rejected by the
 * higher-level API and test gates rather than silently cast or CPU-fallback.
 */

#include "register/op_def_registry.h"

namespace ops {

class Spacing : public OpDef {
public:
    explicit Spacing(const char* name) : OpDef(name)
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

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false);

        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(Spacing);

} // namespace ops