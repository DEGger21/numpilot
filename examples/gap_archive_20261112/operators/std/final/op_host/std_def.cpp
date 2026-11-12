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
 * \file std_def.cpp
 * \brief Std operator definition: float32 input, float32 reduced output, axis attribute.
 */

#include "register/op_def_registry.h"

namespace ops {
// The GE OpType is NumpyStd, not Std: opbuild derives the kernel entry
// name by snake-casing the OpType (NumpyStd -> numpy_std), and an entry
// named `std` would collide with the C++ namespace std in the device
// translation unit.  The public ACLNN interface (aclnnStd /
// aclnnStdGetWorkspaceSize) and the NumPy API identity are unchanged.
class NumpyStd : public OpDef {
public:
    explicit NumpyStd(const char* name) : OpDef(name)
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

        // NumPy-style axis keyword. The observed supported profile uses axis=1.
        // -1 is reserved as the explicit "axis not supplied" sentinel and is
        // rejected by shape/tiling validation rather than silently defaulted.
        this->Attr("axis")
            .AttrType(OPTIONAL)
            .Int(-1);

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
OP_ADD(NumpyStd);
} // namespace ops