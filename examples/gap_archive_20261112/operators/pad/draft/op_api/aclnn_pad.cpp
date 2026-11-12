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
 * @file aclnn_pad.cpp
 * @brief ACLNN L2 API implementation for Pad.
 *
 * Supported bounded contract for this draft:
 *   - input dtype: float32 only
 *   - mode: constant (encoded as 0)
 *   - padWidth: non-negative values, length == 2 * rank
 *   - output dtype: float32, shape == input shape + padWidth
 *
 * The L2 path performs parameter checks, converts non-contiguous input to
 * contiguous form, invokes the L0 Pad operator, and copies the result into the
 * user-provided output tensor. Unsupported modes and dtypes are rejected
 * explicitly; there is no silent host fallback or implicit cast.
 */

#include "aclnn_pad.h"
#include "pad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static constexpr int64_t PAD_MODE_CONSTANT = 0;

static bool CheckNotNull(const aclTensor* self, const aclIntArray* padWidth, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(padWidth, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool IsDtypeSupported(DataType dtype)
{
    return dtype == DataType::DT_FLOAT;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    OP_CHECK(IsDtypeSupported(self->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Pad only supports float32 input in this build, got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    auto selfFormat = self->GetStorageFormat();
    auto outFormat = out->GetStorageFormat();
    OP_CHECK(!(IsPrivateFormat(selfFormat) || IsPrivateFormat(outFormat)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format not supported: self=%d, out=%d",
                     static_cast<int>(selfFormat), static_cast<int>(outFormat)),
             return false);
    return true;
}

static bool CheckMode(int64_t mode)
{
    OP_CHECK(mode == PAD_MODE_CONSTANT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Pad only supports mode=constant (%ld) in this build, got mode=%ld.",
                     PAD_MODE_CONSTANT, mode),
             return false);
    return true;
}

static bool InferOutputShape(const aclTensor* self, const aclIntArray* padWidth, op::Shape& outShape)
{
    size_t rank = self->GetViewShape().GetDimNum();
    if (padWidth->Size() != 2 * rank) {
        return false;
    }

    const int64_t* padData = padWidth->GetData();
    outShape = op::Shape();
    for (size_t i = 0; i < rank; ++i) {
        int64_t before = padData[2 * i];
        int64_t after = padData[2 * i + 1];
        if (before < 0 || after < 0) {
            return false;
        }
        int64_t dim = self->GetViewShape().GetDim(i);
        int64_t outDim = dim + before + after;
        if (outDim < 0) {
            return false;
        }
        outShape.AppendDim(outDim);
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* padWidth, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    op::Shape expected;
    OP_CHECK(InferOutputShape(self, padWidth, expected),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Failed to infer Pad output shape: padWidth_size=%zu, self_rank=%zu.",
                     padWidth->Size(), self->GetViewShape().GetDimNum()),
             return false);

    const auto& outShape = out->GetViewShape();
    OP_CHECK(outShape.GetDimNum() == expected.GetDimNum(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Pad output rank mismatch: expected=%zu, actual=%zu.",
                     expected.GetDimNum(), outShape.GetDimNum()),
             return false);

    for (size_t i = 0; i < expected.GetDimNum(); ++i) {
        OP_CHECK(outShape.GetDim(i) == expected.GetDim(i),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Pad output shape mismatch at dim %zu: expected=%ld, actual=%ld.",
                         i, expected.GetDim(i), outShape.GetDim(i)),
                 return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* padWidth,
                               int64_t mode, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(self, padWidth, out), ACLNN_ERR_PARAM_NULLPTR,
               "CheckNotNull failed");
    CHECK_COND(CheckMode(mode), ACLNN_ERR_PARAM_INVALID,
               "CheckMode failed: mode=%ld", mode);
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: self_dtype=%d, out_dtype=%d",
               static_cast<int>(self->GetDataType()),
               static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: self_format=%d, out_format=%d",
               static_cast<int>(self->GetStorageFormat()),
               static_cast<int>(out->GetStorageFormat()));
    CHECK_COND(CheckShape(self, padWidth, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: self_rank=%zu, out_rank=%zu, padWidth_size=%zu",
               self->GetViewShape().GetDimNum(),
               out->GetViewShape().GetDimNum(),
               padWidth->Size());
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnPadGetWorkspaceSize(
    const aclTensor* self,
    const aclIntArray* padWidth,
    int64_t mode,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnPad, DFX_IN(self, padWidth, mode), DFX_OUT(out));

    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize is null");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "executor is null");

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, padWidth, mode, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    op::Shape expectedOutShape;
    bool inferOk = InferOutputShape(self, padWidth, expectedOutShape);
    CHECK_RET(inferOk, ACLNN_ERR_PARAM_INVALID);

    int64_t outElemCount = 1;
    for (size_t i = 0; i < expectedOutShape.GetDimNum(); ++i) {
        outElemCount *= expectedOutShape.GetDim(i);
    }
    if (outElemCount == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::Pad(selfContiguous, padWidth, mode, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnPad(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPad);
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "executor is null");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}