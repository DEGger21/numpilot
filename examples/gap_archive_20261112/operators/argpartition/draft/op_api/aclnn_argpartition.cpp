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
 * @file aclnn_argpartition.cpp
 * @brief ACLNN L2 API implementation for Argpartition.
 *
 * Supported candidate scope:
 * - self: float32
 * - out: int64, same shape as self
 * - kth and axis follow the numpy.argpartition positional order: (self, kth, axis)
 *
 * The implementation performs explicit parameter validation, contiguous input
 * preparation, L0 operator invocation, and ViewCopy to the caller-provided
 * output tensor. Other dtypes are rejected explicitly; no CPU/NumPy fallback is
 * performed here.
 */

#include "aclnn_argpartition.h"
#include "argpartition.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

#include <cstdint>

using namespace op;

namespace {
constexpr int64_t ARGPARTITION_MAX_SHAPE_RANK = 8;

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK(self->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition only supports float32 input in this candidate, got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);

    OP_CHECK(out->GetDataType() == DataType::DT_INT64,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition output must be int64, got dtype=%d.",
                     static_cast<int>(out->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    const auto selfFormat = self->GetStorageFormat();
    const auto outFormat = out->GetStorageFormat();
    OP_CHECK(!IsPrivateFormat(selfFormat) && !IsPrivateFormat(outFormat),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition does not support private formats: self_format=%d, out_format=%d.",
                     static_cast<int>(selfFormat), static_cast<int>(outFormat)),
             return false);
    return true;
}

static bool SameShape(const op::Shape& lhs, const op::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    const auto selfRank = self->GetViewShape().GetDimNum();
    const auto outRank = out->GetViewShape().GetDimNum();

    OP_CHECK(selfRank > 0 && selfRank <= static_cast<size_t>(ARGPARTITION_MAX_SHAPE_RANK),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition input rank must be in [1, %lld], got %zu.",
                     static_cast<long long>(ARGPARTITION_MAX_SHAPE_RANK), selfRank),
             return false);

    OP_CHECK(outRank == selfRank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition output rank must match input rank: self_rank=%zu, out_rank=%zu.",
                     selfRank, outRank),
             return false);

    OP_CHECK(SameShape(self->GetViewShape(), out->GetViewShape()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition output shape must match input shape."),
             return false);
    return true;
}

static bool NormalizeKthAndAxis(const aclTensor* self, int64_t kth, int64_t axis,
                                int64_t& normalizedKth, int64_t& normalizedAxis)
{
    const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    OP_CHECK(rank > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Argpartition requires rank >= 1."),
             return false);

    OP_CHECK(axis >= -rank && axis < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition axis out of range: axis=%lld, rank=%lld.",
                     static_cast<long long>(axis), static_cast<long long>(rank)),
             return false);

    normalizedAxis = axis < 0 ? axis + rank : axis;
    const int64_t axisSize = self->GetViewShape().GetDim(static_cast<size_t>(normalizedAxis));

    OP_CHECK(axisSize > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition requires the selected axis dimension to be positive, got %lld.",
                     static_cast<long long>(axisSize)),
             return false);

    OP_CHECK(kth >= -axisSize && kth < axisSize,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition kth out of range: kth=%lld, axis_size=%lld.",
                     static_cast<long long>(kth), static_cast<long long>(axisSize)),
             return false);

    // numpy accepts negative kth values. Normalize them before passing the
    // attribute to the Ascend C host/kernel implementation.
    normalizedKth = kth < 0 ? kth + axisSize : kth;
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, int64_t kth, int64_t axis,
                               const aclTensor* out, int64_t& normalizedKth,
                               int64_t& normalizedAxis)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR,
               "Argpartition parameter null check failed.");

    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "Argpartition dtype check failed: self_dtype=%d, out_dtype=%d.",
               static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));

    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "Argpartition format check failed: self_format=%d, out_format=%d.",
               static_cast<int>(self->GetStorageFormat()), static_cast<int>(out->GetStorageFormat()));

    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID,
               "Argpartition shape check failed: self_rank=%zu, out_rank=%zu.",
               self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());

    CHECK_COND(NormalizeKthAndAxis(self, kth, axis, normalizedKth, normalizedAxis),
               ACLNN_ERR_PARAM_INVALID,
               "Argpartition kth/axis check failed: kth=%lld, axis=%lld.",
               static_cast<long long>(kth), static_cast<long long>(axis));

    return ACLNN_SUCCESS;
}
} // namespace

/**
 * @brief First ACLNN stage: validate parameters, prepare input, and create the
 *        deferred Argpartition L0 call.
 */
extern "C" aclnnStatus aclnnArgpartitionGetWorkspaceSize(
    const aclTensor* self,
    int64_t kth,
    int64_t axis,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnArgpartition, DFX_IN(self), DFX_OUT(out));

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    int64_t normalizedKth = 0;
    int64_t normalizedAxis = 0;
    const auto ret = CheckParams(self, kth, axis, out, normalizedKth, normalizedAxis);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult =
        l0op::Argpartition(selfContiguous, normalizedKth, normalizedAxis, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief Second ACLNN stage: launch the deferred Argpartition execution.
 *
 * Error propagation is handled by CommonOpExecutorRun. Synchronization remains
 * owned by the caller/runtime stream contract, matching ACLNN async execution
 * semantics; no implicit host fallback is performed.
 */
extern "C" aclnnStatus aclnnArgpartition(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnArgpartition);

    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}