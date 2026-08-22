/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_def.cpp
 * \brief SmoothL1Loss 算子定义（reduction=mean, beta=1.0 固定语义）
 *
 * 注意：内部注册 OpType 使用全局唯一且不含数字的 NumPilotSmoothMeanLoss。
 * CANN 9.0 的构建期与运行期对 CamelCase OpType 的文件名转换在嵌入数字
 * （如 NumPilotSmoothL1Loss 中的 "1"）处不一致，导致运行期按
 * num_pilot_smooth_l_1_loss.json / num_pilot_smooth_l1loss.json 查找
 * 动态 kernel 配置失败（ACLNN 561103/561108）。改用无数字标识符后，
 * 构建期生成的 num_pilot_smooth_mean_loss.json 与运行期期望路径一致。
 * 公共 ACLNN 符号名（aclnnSmoothL1Loss*）与文件名保持 smooth_l1_loss 不变。
 *
 * NumPy oracle:
 *   error = predict - label
 *   absolute = |error|
 *   bounded = min(absolute, 1.0)
 *   quadratic = bounded * bounded / 2.0
 *   linear = absolute - bounded
 *   y = mean(quadratic + linear)   // 标量输出
 */

#include "register/op_def_registry.h"

namespace ops {
class NumPilotSmoothMeanLoss : public OpDef {
public:
    explicit NumPilotSmoothMeanLoss(const char* name) : OpDef(name)
    {
        // 输入 predict：必选，float32，ND，rank>=1，自动连续化
        this->Input("predict")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        // 输入 label：必选，float32，与 predict 同 shape
        this->Input("label")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        // 输出 y：标量（shape = [1]），float32，mean reduction 结果
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
OP_ADD(NumPilotSmoothMeanLoss);
} // namespace ops