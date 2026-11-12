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
 * @file aclnn_ediff1d.cpp
 * @brief ACLNN L2 API implementation for Ediff1d.
 *
 * Supported scope for this unverified generation stage:
 * - one positional tensor input, no keyword arguments;
 * - float32 input and float32 output only;
 * - NumPy ediff1d semantics over the C-order flattened input;
 * - output length is max(numel(input) - 1, 0).
 *
 * Unsupported dtypes, keyword arguments, and mismatched output shapes are
 * rejected explicitly. Qualification remains pending until the deterministic
 * validator runs the required build, unit, numerical, device, and activation
 * gates.
 */

#include "aclnn_ediff1d.h"
#include "ediff1d.h"

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace {

constexpr int64_t kAclnnMaxShapeRank = 8;

const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool GetShapeNumel(const op::Shape& shape, int64_t& numel)
{
    numel = 1;
    const size_t rank = shape.GetDimNum();
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0) {
            return false;
        }
        if (dim == 0) {
            numel = 0;
            continue;
        }
        if (numel > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        numel *= dim;
    }
    return true;
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
                     "Ediff1d only supports float32 input in this candidate, got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);
    OP_CHECK(out->GetDataType() == self->GetDataType(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Ediff1d output dtype must match input dtype: input=%d, output=%d.",
                     static_cast<int>(self->GetDataType()),
                     static_cast<int>(out->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    const auto selfFormat = self->GetStorageFormat();
    const auto outFormat = out->GetStorageFormat();
    OP_CHECK(!(IsPrivateFormat(selfFormat) || IsPrivateFormat(outFormat)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format is not supported: input=%d, output=%d.",
                     static_cast<int>(selfFormat),
                     static_cast<int>(outFormat)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, kAclnnMaxShapeRank, return false);
    OP_CHECK_MAX_DIM(out, kAclnnMaxShapeRank, return false);

    const auto selfShape = self->GetViewShape();
    const auto outShape = out->GetViewShape();

    OP_CHECK(outShape.GetDimNum() == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Ediff1d output must be rank 1, got rank=%zu.",
                     outShape.GetDimNum()),
             return false);

    int64_t selfNumel = 0;
    OP_CHECK(GetShapeNumel(selfShape, selfNumel),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input shape element count overflow or invalid negative dimension."),
             return false);

    const int64_t expectedLen = (selfNumel > 0) ? (selfNumel - 1) : 0;
    const int64_t outLen = outShape.GetDim(0);
    OP_CHECK(outLen == expectedLen,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Ediff1d output length mismatch: expected %ld, got %ld.",
                     expectedLen,
                     outLen),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: input_dtype=%d, output_dtype=%d.",
               static_cast<int>(self->GetDataType()),
               static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: input_format=%d, output_format=%d.",
               static_cast<int>(self->GetStorageFormat()),
               static_cast<int>(out->GetStorageFormat()));
    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: input_rank=%zu, output_rank=%zu.",
               self->GetViewShape().GetDimNum(),
               out->GetViewShape().GetDimNum());
    return ACLNN_SUCCESS;
}

} // namespace

/**
 * @brief First ACLNN stage: validate parameters, build the executor, and
 *        compute the workspace size required to execute Ediff1d.
 *
 * The user-provided output tensor is validated against NumPy ediff1d shape
 * semantics before any kernel launch is scheduled. Empty outputs are valid
 * when the flattened input has zero or one element.
 */
extern "C" aclnnStatus aclnnEdiff1dGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnEdiff1d, DFX_IN(self), DFX_OUT(out));

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto opResult = l0op::Ediff1d(selfContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief Second ACLNN stage: launch the scheduled Ediff1d kernel.
 *
 * ACLNN execution is asynchronous with respect to the host. The caller owns
 * stream synchronization for numerical checks. Launch errors are surfaced
 * synchronously through CommonOpExecutorRun's aclnnStatus return value.
 */
extern "C" aclnnStatus aclnnEdiff1d(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnEdiff1d);

    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize == 0 || workspace != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}