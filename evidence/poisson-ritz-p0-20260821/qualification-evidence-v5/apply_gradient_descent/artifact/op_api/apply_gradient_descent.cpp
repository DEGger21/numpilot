/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file apply_gradient_descent.cpp
 * @brief ACLNN L0 API 实现 - ApplyGradientDescent
 *
 * 语义（NumPy oracle）: out = var - alpha * delta
 * 参数顺序: var, alpha, delta（3 个位置参数，无关键字参数）
 * dtype 范围: 仅 FLOAT (float32)
 * shape 规则: rank >= 1，输入连续（Contiguous 由 L2 层保证）
 *
 * L0 API 职责：形状推导、Kernel 调度
 * L2 API 职责：参数检查、Contiguous/ViewCopy 处理
 */

#include "apply_gradient_descent.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ApplyGradientDescent);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsAiCoreSupport(const aclTensor* var, const aclTensor* alpha, const aclTensor* delta)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "ApplyGradientDescent not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(var->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
             CheckType(alpha->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
             CheckType(delta->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "ApplyGradientDescent not supported: dtype var=%d, alpha=%d, delta=%d. "
                     "Supported dtype: FLOAT.",
                     static_cast<int>(var->GetDataType()),
                     static_cast<int>(alpha->GetDataType()),
                     static_cast<int>(delta->GetDataType())),
             return false);
    return true;
}

static bool ApplyGradientDescentInferShape(const op::Shape& varShape, op::Shape& outShape)
{
    // 输出形状与 var 一致（alpha 为标量，delta 与 var 同形）
    OP_CHECK(varShape.GetDimNum() >= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "ApplyGradientDescent requires rank >= 1, got rank=%zu.",
                     varShape.GetDimNum()),
             return false);
    outShape = varShape;
    return true;
}

static const aclTensor* ApplyGradientDescentAiCore(const aclTensor* var,
                                                   const aclTensor* alpha,
                                                   const aclTensor* delta,
                                                   const aclTensor* out,
                                                   aclOpExecutor* executor)
{
    L0_DFX(ApplyGradientDescentAiCore, var, alpha, delta, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ApplyGradientDescent,
        OP_INPUT(var, alpha, delta), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ApplyGradientDescentAiCore failed."),
        return nullptr);
    return out;
}

/**
 * @brief L0 API 入口
 *
 * 流程：
 * 1. InferShape      - 输出形状与 var 一致
 * 2. IsAiCoreSupport - 判断执行路径（架构 + dtype）
 * 3. AllocTensor     - 分配输出 Tensor
 * 4. {Op}AiCore      - 调用 Kernel
 */
const aclTensor* ApplyGradientDescent(const aclTensor* var,
                                      const aclTensor* alpha,
                                      const aclTensor* delta,
                                      aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    OP_CHECK(ApplyGradientDescentInferShape(var->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);

    out = executor->AllocTensor(outShape, var->GetDataType());

    OP_CHECK(IsAiCoreSupport(var, alpha, delta),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    return ApplyGradientDescentAiCore(var, alpha, delta, out, executor);
}

} // namespace l0op