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
 * @file aclnn_smooth_l1_loss.cpp
 * @brief ACLNN L2 API 实现 - SmoothL1Loss（mean 归约，beta=1，float32）
 *
 * 语义（NumPy oracle）：
 *   error = predict - label
 *   absolute = |error|
 *   bounded = min(absolute, 1.0)
 *   out = mean(bounded^2 / 2 + (absolute - bounded))
 *
 * 输出为 0 维标量（shape=[]），dtype=float32。
 * 输入仅支持 float32，rank>=1，形状相同。
 */

#include "aclnn_smooth_l1_loss.h"
#include "smooth_l1_loss.h"
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

static bool HasEmptyTensor(const aclTensor* predict, const aclTensor* label)
{
    return predict->IsEmpty() || label->IsEmpty();
}

static bool CheckNotNull(const aclTensor* predict, const aclTensor* label, const aclTensor* out)
{
    OP_CHECK_NULL(predict, return false);
    OP_CHECK_NULL(label, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* predict, const aclTensor* label, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(predict, label->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, predict->GetDataType(), return false);

    OP_CHECK(IsDtypeSupported(predict->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Dtype not supported: dtype=%d. Supported: FLOAT only.",
                     static_cast<int>(predict->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* predict, const aclTensor* label, const aclTensor* out)
{
    auto formatPredict = predict->GetStorageFormat();
    auto formatLabel = label->GetStorageFormat();
    auto formatOut = out->GetStorageFormat();

    OP_CHECK(!(IsPrivateFormat(formatPredict) || IsPrivateFormat(formatLabel) || IsPrivateFormat(formatOut)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format not supported: predict=%d, label=%d, out=%d",
                     static_cast<int>(formatPredict), static_cast<int>(formatLabel),
                     static_cast<int>(formatOut)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* predict, const aclTensor* label, const aclTensor* out)
{
    // rank >= 1 且不超过 ACLNN 支持上限
    OP_CHECK(predict->GetViewShape().GetDimNum() >= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "predict rank must be >= 1."),
             return false);
    OP_CHECK(label->GetViewShape().GetDimNum() >= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "label rank must be >= 1."),
             return false);
    OP_CHECK_MAX_DIM(predict, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(label, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    OP_CHECK(predict->GetViewShape() == label->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "predict and label shapes must be identical."),
             return false);

    // 输出必须是标量（0 维）
    OP_CHECK(out->GetViewShape().GetDimNum() == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "out must be a scalar tensor (rank 0)."),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* predict, const aclTensor* label, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(predict, label, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(predict, label, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: predict_dtype=%d, label_dtype=%d, out_dtype=%d",
               static_cast<int>(predict->GetDataType()), static_cast<int>(label->GetDataType()),
               static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(predict, label, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: predict_format=%d, label_format=%d, out_format=%d",
               static_cast<int>(predict->GetStorageFormat()), static_cast<int>(label->GetStorageFormat()),
               static_cast<int>(out->GetStorageFormat()));
    CHECK_COND(CheckShape(predict, label, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: predict_dim=%zu, label_dim=%zu, out_dim=%zu",
               predict->GetViewShape().GetDimNum(), label->GetViewShape().GetDimNum(),
               out->GetViewShape().GetDimNum());
    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：计算 workspace 大小
 *
 * 标准流程：
 * 1. CREATE_EXECUTOR()         - 创建执行器
 * 2. CheckParams()             - 参数检查
 * 3. HasEmptyTensor()          - 空 Tensor 快速返回
 * 4. Contiguous()              - 非连续 Tensor 转换（保证 Kernel 输入连续）
 * 5. l0op::SmoothL1Loss()      - 调用 L0 算子
 * 6. ViewCopy()                - 输出非连续处理
 * 7. GetWorkspaceSize()        - 获取 workspace 大小
 */
extern "C" aclnnStatus aclnnSmoothL1LossGetWorkspaceSize(
    const aclTensor* predict,
    const aclTensor* label,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSmoothL1Loss, DFX_IN(predict, label), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(predict, label, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(predict, label)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto predictContiguous = l0op::Contiguous(predict, uniqueExecutor.get());
    CHECK_RET(predictContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto labelContiguous = l0op::Contiguous(label, uniqueExecutor.get());
    CHECK_RET(labelContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::SmoothL1Loss(predictContiguous, labelContiguous, uniqueExecutor.get());
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
extern "C" aclnnStatus aclnnSmoothL1Loss(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSmoothL1Loss);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}