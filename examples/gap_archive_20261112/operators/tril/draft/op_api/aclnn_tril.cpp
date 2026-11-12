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
 * @file aclnn_tril.cpp
 * @brief ACLNN L2 API implementation for Tril.
 *
 * Public mapping:
 *   asnumpy.tril -> Python wrapper -> pybind tril -> native adapter -> aclnnTrilGetWorkspaceSize/aclnnTril
 *   -> l0op::Tril -> Tril Ascend C kernel.
 *
 * Supported scope for this draft:
 *   input dtype: float32 only
 *   input rank: 2 only
 *   output dtype: float32
 *   output shape: same as input
 *   semantics: lower triangle with k == 0, matching numpy.tril(a)
 */

#include "aclnn_tril.h"
#include "tril.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace {

constexpr size_t TRIL_SUPPORTED_RANK = 2;

static const std::initializer_list<op::DataType> TRIL_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, TRIL_DTYPE_SUPPORT_LIST);
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);

    OP_CHECK(IsDtypeSupported(self->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Tril only supports float32 in this draft. Got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);

    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    const auto selfFormat = self->GetStorageFormat();
    const auto outFormat = out->GetStorageFormat();

    OP_CHECK(!(IsPrivateFormat(selfFormat) || IsPrivateFormat(outFormat)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Tril does not support private formats. self_format=%d, out_format=%d.",
                     static_cast<int>(selfFormat), static_cast<int>(outFormat)),
             return false);

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    const auto selfShape = self->GetViewShape();
    const auto outShape = out->GetViewShape();

    OP_CHECK(selfShape.GetDimNum() == TRIL_SUPPORTED_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Tril only supports rank-2 inputs in this draft. Got self_dim=%zu.",
                     selfShape.GetDimNum()),
             return false);

    OP_CHECK(outShape.GetDimNum() == TRIL_SUPPORTED_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Tril expects a rank-2 output tensor. Got out_dim=%zu.",
                     outShape.GetDimNum()),
             return false);

    for (size_t i = 0; i < TRIL_SUPPORTED_RANK; ++i) {
        OP_CHECK(selfShape.GetDim(i) == outShape.GetDim(i),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Tril output shape must match input shape. Axis=%zu, self_dim=%lld, out_dim=%lld.",
                         i, static_cast<long long>(selfShape.GetDim(i)),
                         static_cast<long long>(outShape.GetDim(i))),
                 return false);
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");

    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: self_dtype=%d, out_dtype=%d",
               static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));

    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: self_format=%d, out_format=%d",
               static_cast<int>(self->GetStorageFormat()), static_cast<int>(out->GetStorageFormat()));

    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: self_dim=%zu, out_dim=%zu",
               self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());

    return ACLNN_SUCCESS;
}

} // namespace

/**
 * @brief First-stage ACLNN API: validate parameters and build the executor.
 *
 * @param self [in] input tensor, float32, rank 2
 * @param out [in] output tensor, float32, same shape as self
 * @param workspaceSize [out] required workspace size in bytes
 * @param executor [out] created ACLNN executor
 * @return aclnnStatus
 */
extern "C" aclnnStatus aclnnTrilGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnTril, DFX_IN(self), DFX_OUT(out));

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclnnStatus checkStatus = CheckParams(self, out);
    CHECK_RET(checkStatus == ACLNN_SUCCESS, checkStatus);

    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* trilResult = l0op::Tril(selfContiguous, uniqueExecutor.get());
    CHECK_RET(trilResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(trilResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief Second-stage ACLNN API: enqueue execution on the caller-provided stream.
 *
 * @param workspace [in] workspace memory
 * @param workspaceSize [in] workspace size in bytes
 * @param executor [in] prepared ACLNN executor
 * @param stream [in] ACL stream
 * @return aclnnStatus
 */
extern "C" aclnnStatus aclnnTril(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTril);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}