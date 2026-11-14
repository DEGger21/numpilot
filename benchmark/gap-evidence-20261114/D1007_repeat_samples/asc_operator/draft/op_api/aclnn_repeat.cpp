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
 * @file aclnn_repeat.cpp
 * @brief ACLNN L2 API implementation for Repeat.
 *
 * Supported bounded scope for this unverified draft:
 * - float32 input/output only
 * - scalar non-negative repeats
 * - non-empty rank-1 or higher input with a valid axis in [-rank, rank - 1]
 * - axis=None / scalar-input flattening is explicitly rejected by the ACLNN entry
 *
 * The public NumPy argument order is preserved:
 *   repeat(self, repeats, axis, out)
 */

#include "aclnn_repeat.h"
#include "repeat.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

#include <cstdint>
#include <initializer_list>

using namespace op;

namespace {
constexpr int64_t ACLNN_MAX_SHAPE_RANK = 8;

const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);

    OP_CHECK(IsDtypeSupported(self->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Repeat only supports DT_FLOAT in this bounded draft, got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);
    return true;
}

bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    const auto selfFormat = self->GetStorageFormat();
    const auto outFormat = out->GetStorageFormat();

    OP_CHECK(!(IsPrivateFormat(selfFormat) || IsPrivateFormat(outFormat)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Repeat does not support private formats: self=%d, out=%d.",
                     static_cast<int>(selfFormat), static_cast<int>(outFormat)),
             return false);
    return true;
}

bool CheckRepeats(int64_t repeats)
{
    OP_CHECK(repeats >= 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Repeat requires repeats >= 0, got repeats=%lld.",
                     static_cast<long long>(repeats)),
             return false);
    return true;
}

bool CheckAxis(const aclTensor* self, int64_t axis)
{
    const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());

    OP_CHECK(rank > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Repeat axis-based mode requires rank > 0 input; axis=None flattening is not supported."),
             return false);

    OP_CHECK(axis >= -rank && axis < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Repeat axis=%lld is out of range for input rank=%lld.",
                     static_cast<long long>(axis), static_cast<long long>(rank)),
             return false);
    return true;
}

bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

aclnnStatus CheckParams(const aclTensor* self, int64_t repeats, int64_t axis, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed.");
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: self_dtype=%d, out_dtype=%d.",
               static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: self_format=%d, out_format=%d.",
               static_cast<int>(self->GetStorageFormat()), static_cast<int>(out->GetStorageFormat()));
    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: self_dim=%zu, out_dim=%zu.",
               self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());
    CHECK_COND(CheckRepeats(repeats), ACLNN_ERR_PARAM_INVALID,
               "CheckRepeats failed: repeats=%lld.", static_cast<long long>(repeats));
    CHECK_COND(CheckAxis(self, axis), ACLNN_ERR_PARAM_INVALID,
               "CheckAxis failed: axis=%lld.", static_cast<long long>(axis));
    return ACLNN_SUCCESS;
}

int64_t NormalizeAxis(const aclTensor* self, int64_t axis)
{
    const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    return axis < 0 ? axis + rank : axis;
}
} // namespace

extern "C" aclnnStatus aclnnRepeatGetWorkspaceSize(
    const aclTensor* self,
    int64_t repeats,
    int64_t axis,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRepeat, DFX_IN(self), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const auto ret = CheckParams(self, repeats, axis, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // A zero repeat count or an empty input produces no elements to write.
    if (self->IsEmpty() || repeats == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const int64_t normalizedAxis = NormalizeAxis(self, axis);

    const aclTensor* selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult =
        l0op::Repeat(selfContiguous, repeats, normalizedAxis, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnRepeat(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRepeat);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}