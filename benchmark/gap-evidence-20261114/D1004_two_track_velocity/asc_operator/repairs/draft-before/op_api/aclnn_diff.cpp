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
 * @file aclnn_diff.cpp
 * @brief ACLNN L2 API implementation for Diff.
 *
 * This is an unverified draft generated for the numpy.diff gap. The admitted
 * runtime profile is float32, C-contiguous input. The implementation rejects
 * non-float32 inputs and invalid axes explicitly instead of casting or falling
 * back to the host.
 *
 * ACLNN two-stage contract:
 * 1. aclnnDiffGetWorkspaceSize - validate parameters and build the executor.
 * 2. aclnnDiff                 - launch the operator and synchronize the stream.
 */

#include "aclnn_diff.h"

#include <cstdint>

#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "diff.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace {

constexpr int64_t kDiffMaxShapeRank = 8;

static bool CheckNotNull(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{
    OP_CHECK(self->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Diff only supports float32 input in this candidate, but got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);
    OP_CHECK(out->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Diff only supports float32 output in this candidate, but got dtype=%d.",
                     static_cast<int>(out->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out)
{
    const auto selfFormat = self->GetStorageFormat();
    const auto outFormat = out->GetStorageFormat();

    OP_CHECK(!IsPrivateFormat(selfFormat) && !IsPrivateFormat(outFormat),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private formats are not supported: self_format=%d, out_format=%d.",
                     static_cast<int>(selfFormat), static_cast<int>(outFormat)),
             return false);
    return true;
}

static bool NormalizeAxis(int64_t axis, size_t rank, int64_t *normalizedAxis)
{
    if (normalizedAxis == nullptr || rank == 0) {
        return false;
    }

    const int64_t signedRank = static_cast<int64_t>(rank);
    if (axis < -signedRank || axis >= signedRank) {
        return false;
    }

    *normalizedAxis = (axis < 0) ? (axis + signedRank) : axis;
    return true;
}

static bool CheckShape(const aclTensor *self, int64_t axis, const aclTensor *out,
                       int64_t *normalizedAxis)
{
    if (normalizedAxis == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Diff normalized axis output pointer is null.");
        return false;
    }

    const auto selfShape = self->GetViewShape();
    const auto outShape = out->GetViewShape();
    const size_t rank = selfShape.GetDimNum();

    OP_CHECK(rank > 0 && rank <= static_cast<size_t>(kDiffMaxShapeRank),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Diff input rank must be in [1, %lld], but got %zu.",
                     static_cast<long long>(kDiffMaxShapeRank), rank),
             return false);
    OP_CHECK(outShape.GetDimNum() == rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Diff output rank must equal input rank %zu, but got %zu.",
                     rank, outShape.GetDimNum()),
             return false);

    int64_t localNormalizedAxis = 0;
    OP_CHECK(NormalizeAxis(axis, rank, &localNormalizedAxis),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Diff axis %lld is out of range for input rank %zu.",
                     static_cast<long long>(axis), rank),
             return false);

    for (size_t i = 0; i < rank; ++i) {
        const int64_t selfDim = selfShape.GetDim(i);
        int64_t expectedDim = selfDim;
        if (i == static_cast<size_t>(localNormalizedAxis)) {
            expectedDim = (selfDim > 0) ? (selfDim - 1) : 0;
        }

        const int64_t actualDim = outShape.GetDim(i);
        OP_CHECK(actualDim == expectedDim,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Diff output shape mismatch at dim %zu: expected %lld, got %lld.",
                         i, static_cast<long long>(expectedDim),
                         static_cast<long long>(actualDim)),
                 return false);
    }

    *normalizedAxis = localNormalizedAxis;
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t axis, const aclTensor *out,
                               int64_t *normalizedAxis)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR,
               "Diff parameter check failed: null tensor.");
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "Diff parameter check failed: unsupported dtype.");
    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "Diff parameter check failed: unsupported format.");
    CHECK_COND(CheckShape(self, axis, out, normalizedAxis), ACLNN_ERR_PARAM_INVALID,
               "Diff parameter check failed: invalid shape or axis.");
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnDiffGetWorkspaceSize(
    const aclTensor *self,
    int64_t axis,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;

    L2_DFX_PHASE_1(aclnnDiff, DFX_IN(self, axis), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    int64_t normalizedAxis = 0;
    const aclnnStatus checkStatus = CheckParams(self, axis, out, &normalizedAxis);
    CHECK_RET(checkStatus == ACLNN_SUCCESS, checkStatus);

    // An empty result has no elements to write. This occurs naturally for
    // NumPy-compatible cases such as an axis dimension of length 0 or 1.
    if (out->IsEmpty()) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor *selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *opResult = l0op::Diff(selfContiguous, normalizedAxis, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnDiff(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDiff);

    if (executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (workspaceSize > 0 && workspace == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    const aclnnStatus runStatus = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (runStatus != ACLNN_SUCCESS) {
        return runStatus;
    }

    // This draft exposes an explicitly synchronized execution stage so callers
    // cannot observe partially written Diff outputs. The public adapter and
    // tests may also synchronize, but this keeps the ACLNN execution boundary
    // deterministic for unverified numerical validation.
    if (stream != nullptr) {
        const aclError syncStatus = aclrtSynchronizeStream(stream);
        if (syncStatus != ACL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                    "aclnnDiff stream synchronization failed, aclError=%d.",
                    static_cast<int32_t>(syncStatus));
            return static_cast<aclnnStatus>(syncStatus);
        }
    }

    return ACLNN_SUCCESS;
}