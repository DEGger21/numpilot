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
 * \file apply_gradient_descent_def.cpp
 * \brief ApplyGradientDescent 算子定义，声明输入输出和算子配置
 *        语义（NumPy oracle）: y = var - alpha * delta
 *        输入顺序固定为: var, alpha, delta；输出: y；仅支持 float32。
 */

#include "register/op_def_registry.h"

namespace ops {
class ApplyGradientDescent : public OpDef {
public:
    explicit ApplyGradientDescent(const char* name) : OpDef(name)
    {
        this->Input("var")                                        // 输入 var 定义
            .ParamType(REQUIRED)                                  // 必选输入
            .DataType({ge::DT_FLOAT})                             // 仅支持 float32
            .Format({ge::FORMAT_ND})                              // ND 格式
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();                                    // 内存自动连续化
        this->Input("alpha")                                      // 输入 alpha 定义（标量，shape 为 {1} 或与 var 广播一致）
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("delta")                                      // 输入 delta 定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")                                         // 输出 y 定义: y = var - alpha * delta
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
OP_ADD(ApplyGradientDescent); // 添加算子信息库
} // namespace ops