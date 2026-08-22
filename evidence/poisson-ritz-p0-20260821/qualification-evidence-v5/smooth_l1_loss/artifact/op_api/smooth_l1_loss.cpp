/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file smooth_l1_loss.cpp
 * @brief ACLNN L0 API 实现 - SmoothL1Loss（mean 归约，beta=1，float32）
 *
 * 语义（NumPy oracle）：
 *   error = predict - label
 *   absolute = |error|
 *   bounded = min(absolute, 1.0)
 *   loss = mean(bounded^2 / 2 + (absolute - bounded))
 * 输出为 0 维标量张量（float32）。
 *
 * 注意：内部注册/调度 OpType 使用全局唯一且不含数字的
 * NumPilotSmoothMeanLoss。CANN 9.0 的构建期与运行期对含嵌入数字的
 * CamelCase OpType（如 NumPilotSmoothL1Loss）生成的 JSON 文件名不一致
 * （num_pilot_smooth_l1_loss.json vs num_pilot_smooth_l_1_loss.json /
 * num_pilot_smooth_l1loss.json），导致 561103/561108
 * "Get op json info failed / Parse dynamic kernel config fail"。
 * 使用无数字标识符后两侧转换结果一致。对外 C 符号（aclnnSmoothL1Loss*）
 * 与语义保持不变。
 *
 * L0 API 职责：形状推导、Kernel 调度
 * L2 API 职责：参数检查、Contiguous/ViewCopy 处理（见 aclnn_smooth_l1_loss.cpp）
 */

#include "smooth_l1_loss.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(NumPilotSmoothMeanLoss);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsAiCoreSupport(const aclTensor* predict, const aclTensor* label)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SmoothL1Loss not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(predict->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
             CheckType(label->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SmoothL1Loss not supported: dtype predict=%d, label=%d. Supported dtype: FLOAT.",
                     static_cast<int>(predict->GetDataType()),
                     static_cast<int>(label->GetDataType())),
             return false);
    return true;
}

static bool SmoothL1LossInferShape(const op::Shape& predictShape, const op::Shape& labelShape, op::Shape& outShape)
{
    // 输入必须同形（无广播），输出为 mean 归约后的 0 维标量。
    OP_CHECK(predictShape == labelShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SmoothL1Loss requires predict and label to have identical shapes."),
             return false);
    OP_CHECK(predictShape.GetDimNum() >= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SmoothL1Loss requires input rank >= 1."),
             return false);
    outShape = Shape(); // 0-dim scalar
    return true;
}

static const aclTensor* SmoothL1LossAiCore(const aclTensor* predict, const aclTensor* label,
                                           const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(SmoothL1LossAiCore, predict, label, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(NumPilotSmoothMeanLoss,
        OP_INPUT(predict, label), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SmoothL1LossAiCore failed."),
        return nullptr);
    return out;
}

/**
 * @brief L0 API 入口
 *
 * 流程：
 * 1. InferShape      - 形状推导（同形输入，标量输出）
 * 2. IsAiCoreSupport - 判断执行路径
 * 3. AllocTensor     - 分配输出 Tensor（0 维 float32）
 * 4. SmoothL1LossAiCore - 调用 Kernel
 */
const aclTensor* SmoothL1Loss(const aclTensor* predict, const aclTensor* label, aclOpExecutor* executor)
{
    const aclTensor* out = nullptr;

    Shape outShape;
    OP_CHECK(SmoothL1LossInferShape(predict->GetViewShape(), label->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);

    out = executor->AllocTensor(outShape, predict->GetDataType());
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor failed for SmoothL1Loss output."),
             return nullptr);

    OP_CHECK(IsAiCoreSupport(predict, label),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    return SmoothL1LossAiCore(predict, label, out, executor);
}

} // namespace l0op