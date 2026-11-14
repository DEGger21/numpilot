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
 * @file aclnn_roll.cpp
 * @brief ACLNN L2 API implementation for Roll.
 *
 * Public mapping:
 *   asnumpy.roll -> Python wrapper -> pybind roll -> l0op::Roll
 *   -> aclnnRollGetWorkspaceSize / aclnnRoll
 *
 * This draft admits only the observed float32 tensor profile with an explicit
 * in-range axis. Unsupported dtypes, private formats, shape mismatches, and
 * invalid axes are rejected explicitly; there is no implicit cast or host
 * fallback.
 */

#include "aclnn_roll.h"
#include "roll.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

#include <cstdint>

using namespace op;

#ifndef ACLNN_MAX_SHAPE_RANK
#define ACLNN_MAX_SHAPE_RANK 8
#endif

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool HasEmptyTensor(const aclTensor* self, const aclTensor* out)
{
    return self->IsEmpty() || out->IsEmpty();
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK(IsDtypeSupported(self->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Roll only supports FLOAT32 input in this draft, got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    const auto selfFormat = self->GetStorageFormat();
    const auto outFormat = out->GetStorageFormat();
    OP_CHECK(!(IsPrivateFormat(selfFormat) || IsPrivateFormat(outFormat)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Roll does not support private formats: self_format=%d, out_format=%d.",
                     static_cast<int>(selfFormat), static_cast<int>(outFormat)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    const auto& selfShape = self->GetViewShape();
    const auto& outShape = out->GetViewShape();
    OP_CHECK(selfShape.GetDimNum() == outShape.GetDimNum(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Roll input/output rank mismatch: self_rank=%zu, out_rank=%zu.",
                     selfShape.GetDimNum(), outShape.GetDimNum()),
             return false);

    for (size_t i = 0; i < selfShape.GetDimNum(); ++i) {
        OP_CHECK(selfShape.GetDim(i) == outShape.GetDim(i),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Roll input/output shape mismatch at dim %zu: self_dim=%lld, out_dim=%lld.",
                         i, static_cast<long long>(selfShape.GetDim(i)),
                         static_cast<long long>(outShape.GetDim(i))),
                 return false);
    }
    return true;
}

static bool CheckAxis(const aclTensor* self, int64_t axis)
{
    const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    OP_CHECK(rank > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Roll requires a non-scalar tensor when axis is specified."),
             return false);
    OP_CHECK(axis >= -rank && axis < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Roll axis out of range: axis=%lld, rank=%lld.",
                     static_cast<long long>(axis), static_cast<long long>(rank)),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, int64_t axis, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR,
               "Roll CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "Roll CheckDtypeValid failed: self_dtype=%d, out_dtype=%d",
               static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "Roll CheckFormat failed: self_format=%d, out_format=%d",
               static_cast<int>(self->GetStorageFormat()),
               static_cast<int>(out->GetStorageFormat()));
    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID,
               "Roll CheckShape failed: self_rank=%zu, out_rank=%zu",
               self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());
    CHECK_COND(CheckAxis(self, axis), ACLNN_ERR_PARAM_INVALID,
               "Roll CheckAxis failed: axis=%lld",
               static_cast<long long>(axis));
    return ACLNN_SUCCESS;
}

/**
 * @brief First-stage ACLNN interface: validate parameters and build executor.
 */
extern "C" aclnnStatus aclnnRollGetWorkspaceSize(
    const aclTensor* self,
    int64_t shift,
    int64_t axis,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRoll, DFX_IN(self, shift, axis), DFX_OUT(out));

    CHECK_RET(workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, axis, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    int64_t normalizedAxis = axis;
    const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    if (normalizedAxis < 0) {
        normalizedAxis += rank;
    }

    if (HasEmptyTensor(self, out)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::Roll(
        selfContiguous, shift, normalizedAxis, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief Second-stage ACLNN interface: execute the scheduled Roll operator.
 */
extern "C" aclnnStatus aclnnRoll(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRoll);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}