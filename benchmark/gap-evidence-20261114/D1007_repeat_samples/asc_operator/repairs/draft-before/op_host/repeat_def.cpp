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
 * \file repeat_def.cpp
 * \brief Repeat operator definition. Declares the single float32 tensor input, the single float32 tensor
 *        output, and the repeat count / axis attributes used by shape inference, tiling and the kernel.
 */
#include "register/op_def_registry.h"

namespace ops {
class Repeat : public OpDef {
public:
    explicit Repeat(const char* name) : OpDef(name)
    {
        // NumPy repeat(data, repeats, axis=...)
        // data -> x; the scalar repeats and axis values are carried as operator attributes so that
        // host-side shape inference can compute the output shape without reading device tensor values.
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

        // Attribute order is part of the host API: index 0 is repeats, index 1 is axis.
        this->Attr("repeats").Int();
        this->Attr("axis").Int();

        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);

        this->ExtendCfgInfo("opFile.value", "repeat");
    }
};
OP_ADD(Repeat);
} // namespace ops