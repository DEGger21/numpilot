/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_apply_gradient_descent.cpp
 * @brief ACLNN L2 API 实现 - ApplyGradientDescent
 *
 * 语义（NumPy oracle）: out = var - alpha * delta
 *
 * 两段式接口：
 * 1. aclnnApplyGradientDescentGetWorkspaceSize - 参数检查、Contiguous、调用 L0 算子
 * 2. aclnnApplyGradientDescent - 执行计算
 */

#include "aclnn_apply_gradient_descent.h"
#include "apply_gradient_descent.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool HasEmptyTensor(const aclTensor* var, const aclTensor* alpha, const aclTensor* delta)
{
    return var->IsEmpty() || alpha->IsEmpty() || delta->IsEmpty();
}

static bool CheckNotNull(const aclTensor* var, const aclTensor* alpha,
                         const aclTensor* delta, const aclTensor* out)
{
    OP_CHECK_NULL(var, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(delta, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* var, const aclTensor* alpha,
                            const aclTensor* delta, const aclTensor* out)
{
    OP_CHECK(IsDtypeSupported(var->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Dtype not supported: var dtype=%d. Supported: FLOAT only.",
                     static_cast<int>(var->GetDataType())),
             return false);
    OP_CHECK_DTYPE_NOT_MATCH(alpha, var->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(delta, var->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, var->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensor* var, const aclTensor* alpha,
                        const aclTensor* delta, const aclTensor* out)
{
    OP_CHECK(!(IsPrivateFormat(var->GetStorageFormat()) ||
              IsPrivateFormat(alpha->GetStorageFormat()) ||
              IsPrivateFormat(delta->GetStorageFormat()) ||
              IsPrivateFormat(out->GetStorageFormat())),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format not supported: var=%d, alpha=%d, delta=%d, out=%d",
                     static_cast<int>(var->GetStorageFormat()),
                     static_cast<int>(alpha->GetStorageFormat()),
                     static_cast<int>(delta->GetStorageFormat()),
                     static_cast<int>(out->GetStorageFormat())),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* var, const aclTensor* alpha,
                       const aclTensor* delta, const aclTensor* out)
{
    // 形状规则：rank >= 1，var/delta/out 同形，alpha 为标量（rank 0 或 1 元素）
    OP_CHECK(var->GetViewShape().GetDimNum() >= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "var rank must be >= 1, got rank=%zu.",
                     var->GetViewShape().GetDimNum()),
             return false);
    OP_CHECK(delta->GetViewShape().GetDimNum() >= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "delta rank must be >= 1, got rank=%zu.",
                     delta->GetViewShape().GetDimNum()),
             return false);
    OP_CHECK_MAX_DIM(var, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(delta, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    OP_CHECK(var->GetViewShape() == delta->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "var and delta shapes must match for ApplyGradientDescent."),
             return false);
    OP_CHECK(var->GetViewShape() == out->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "var and out shapes must match for ApplyGradientDescent."),
             return false);
    OP_CHECK(alpha->Numel() == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "alpha must be a scalar tensor, got numel=%ld.",
                     static_cast<long>(alpha->Numel())),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* var, const aclTensor* alpha,
                               const aclTensor* delta, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(var, alpha, delta, out), ACLNN_ERR_PARAM_NULLPTR,
               "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(var, alpha, delta, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: var_dtype=%d, alpha_dtype=%d, delta_dtype=%d, out_dtype=%d",
               static_cast<int>(var->GetDataType()),
               static_cast<int>(alpha->GetDataType()),
               static_cast<int>(delta->GetDataType()),
               static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(var, alpha, delta, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed");
    CHECK_COND(CheckShape(var, alpha, delta, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed");
    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：计算 workspace 大小
 *
 * 流程：
 * 1. CREATE_EXECUTOR()   - 创建执行器
 * 2. CheckParams()       - 参数检查
 * 3. HasEmptyTensor()    - 空 Tensor 快速返回
 * 4. Contiguous()        - 非连续 Tensor 转换
 * 5. l0op::ApplyGradientDescent() - 调用 L0 算子
 * 6. ViewCopy()          - 输出非连续处理
 * 7. GetWorkspaceSize()  - 获取 workspace 大小
 */
extern "C" aclnnStatus aclnnApplyGradientDescentGetWorkspaceSize(
    const aclTensor* var,
    const aclTensor* alpha,
    const aclTensor* delta,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnApplyGradientDescent, DFX_IN(var, alpha, delta), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(var, alpha, delta, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(var, alpha, delta)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto varContiguous = l0op::Contiguous(var, uniqueExecutor.get());
    CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto alphaContiguous = l0op::Contiguous(alpha, uniqueExecutor.get());
    CHECK_RET(alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto deltaContiguous = l0op::Contiguous(delta, uniqueExecutor.get());
    CHECK_RET(deltaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::ApplyGradientDescent(varContiguous, alphaContiguous,
                                                           deltaContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief 第二段接口：执行计算
 */
extern "C" aclnnStatus aclnnApplyGradientDescent(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnApplyGradientDescent);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}