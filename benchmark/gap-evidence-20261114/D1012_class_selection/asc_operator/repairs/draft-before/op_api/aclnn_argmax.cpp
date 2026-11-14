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
 * @file aclnn_argmax.cpp
 * @brief ACLNN L2 API implementation for Argmax.
 *
 * Public mapping:
 *   asnumpy.argmax
 *     -> aclnnArgmaxGetWorkspaceSize / aclnnArgmax
 *     -> l0op::Argmax
 *     -> kernel entry Argmax
 *
 * Supported candidate scope for this generation:
 *   - self:  float32, rank 2, C-contiguous or made contiguous here
 *   - axis:  1 (the observed NumPy callsite parameter)
 *   - out:   int64, rank 1 with shape [self.shape[0]]
 *
 * Other dtypes, ranks, axes, and malformed output descriptions are rejected
 * explicitly. No host fallback and no implicit dtype conversion are performed.
 */

#include "aclnn_argmax.h"
#include "argmax.h"
#include "acl/acl.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

#include <cstddef>
#include <cstdint>

using namespace op;

namespace {

constexpr std::size_t ARGMAX_SUPPORTED_RANK = 2U;
constexpr std::size_t ARGMAX_OUTPUT_RANK = 1U;
constexpr int64_t ARGMAX_SUPPORTED_AXIS = 1;

static bool CheckNotNull(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_MATCH(self, DataType::DT_FLOAT, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, DataType::DT_INT64, return false);
    return true;
}

static bool CheckFormatValid(const aclTensor *self, const aclTensor *out)
{
    const auto selfFormat = self->GetStorageFormat();
    const auto outFormat = out->GetStorageFormat();
    OP_CHECK(!IsPrivateFormat(selfFormat) && !IsPrivateFormat(outFormat),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax does not support private formats: self_format=%d, out_format=%d.",
                     static_cast<int>(selfFormat), static_cast<int>(outFormat)),
             return false);
    return true;
}

static bool CheckShapeValid(const aclTensor *self, int64_t axis, const aclTensor *out)
{
    const auto selfShape = self->GetViewShape();
    const auto outShape = out->GetViewShape();

    OP_CHECK(selfShape.GetDimNum() == ARGMAX_SUPPORTED_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax supports only rank-2 input in this candidate, got rank=%zu.",
                     selfShape.GetDimNum()),
             return false);

    OP_CHECK(axis == ARGMAX_SUPPORTED_AXIS,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax supports only axis=1 in this candidate, got axis=%lld.",
                     static_cast<long long>(axis)),
             return false);

    OP_CHECK(selfShape.GetDim(0) > 0 && selfShape.GetDim(1) > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax does not support empty input in this candidate, got shape=[%lld, %lld].",
                     static_cast<long long>(selfShape.GetDim(0)),
                     static_cast<long long>(selfShape.GetDim(1))),
             return false);

    OP_CHECK(outShape.GetDimNum() == ARGMAX_OUTPUT_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax output must be rank 1 for axis=1, got rank=%zu.",
                     outShape.GetDimNum()),
             return false);

    OP_CHECK(outShape.GetDim(0) == selfShape.GetDim(0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax output shape mismatch for axis=1: expected [%lld], got [%lld].",
                     static_cast<long long>(selfShape.GetDim(0)),
                     static_cast<long long>(outShape.GetDim(0))),
             return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t axis, const aclTensor *out)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR,
               "Argmax parameter is null: self=%p, out=%p.",
               static_cast<const void *>(self), static_cast<const void *>(out));

    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "Argmax dtype mismatch: self_dtype=%d, out_dtype=%d. Expected float32 input and int64 output.",
               static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));

    CHECK_COND(CheckFormatValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "Argmax format check failed: self_format=%d, out_format=%d.",
               static_cast<int>(self->GetStorageFormat()),
               static_cast<int>(out->GetStorageFormat()));

    CHECK_COND(CheckShapeValid(self, axis, out), ACLNN_ERR_PARAM_INVALID,
               "Argmax shape/axis check failed: axis=%lld, self_rank=%zu, out_rank=%zu.",
               static_cast<long long>(axis),
               self->GetViewShape().GetDimNum(),
               out->GetViewShape().GetDimNum());

    return ACLNN_SUCCESS;
}

} // namespace

/**
 * @brief First ACLNN stage: validate parameters and build the execution plan.
 *
 * The input is made contiguous before the L0 Argmax dispatch. The L0 operator
 * allocates the int64 result tensor, and ViewCopy materializes the result into
 * the caller-provided output tensor.
 */
extern "C" aclnnStatus aclnnArgmaxGetWorkspaceSize(
    const aclTensor *self,
    int64_t axis,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnArgmax, DFX_IN(self, axis), DFX_OUT(out));

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = 0U;
    *executor = nullptr;

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclnnStatus checkRet = CheckParams(self, axis, out);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    const aclTensor *selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *opResult = l0op::Argmax(selfContiguous, axis, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief Second ACLNN stage: launch the prepared executor and wait for completion.
 *
 * The synchronization below is intentionally explicit for this accuracy-first
 * candidate. Callers may still synchronize the same stream again without any
 * correctness change.
 */
extern "C" aclnnStatus aclnnArgmax(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnArgmax);

    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (workspaceSize > 0U) {
        CHECK_RET(workspace != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    const aclnnStatus runRet = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    CHECK_RET(runRet == ACLNN_SUCCESS, runRet);

    if (stream != nullptr) {
        const aclError syncRet = aclrtSynchronizeStream(stream);
        if (syncRet != ACL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "aclnnArgmax aclrtSynchronizeStream failed, ret=%d.",
                    static_cast<int>(syncRet));
            return ACLNN_ERR_INNER;
        }
    }

    return ACLNN_SUCCESS;
}