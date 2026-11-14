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
 * @file aclnn_spacing.cpp
 * @brief ACLNN L2 API implementation for Spacing.
 *
 * The public ACLNN contract is the standard two-stage interface:
 *   1. aclnnSpacingGetWorkspaceSize - validate, build the executor, and return workspace size.
 *   2. aclnnSpacing                  - submit the queued kernel on the caller-provided stream.
 *
 * This candidate intentionally supports only the observed float32 input/output profile.
 * Other dtypes are rejected explicitly rather than silently cast or executed on the host.
 */

#include "aclnn_spacing.h"
#include "spacing.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

#include <cstddef>
#include <initializer_list>

using namespace op;

namespace {

constexpr int64_t ACLNN_SPACING_MAX_SHAPE_RANK = 8;

const std::initializer_list<op::DataType> SPACING_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

bool IsSpacingDtypeSupported(DataType dtype)
{
    return CheckType(dtype, SPACING_AICORE_DTYPE_SUPPORT_LIST);
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

    OP_CHECK(IsSpacingDtypeSupported(self->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Spacing only supports float32 input/output in this candidate; input dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);
    return true;
}

bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    const auto inputFormat = self->GetStorageFormat();
    const auto outputFormat = out->GetStorageFormat();

    OP_CHECK(!(IsPrivateFormat(inputFormat) || IsPrivateFormat(outputFormat)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Spacing does not support private storage formats: input=%d, output=%d.",
                     static_cast<int>(inputFormat), static_cast<int>(outputFormat)),
             return false);
    return true;
}

bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK(self->GetViewShape().GetDimNum() <= ACLNN_SPACING_MAX_SHAPE_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Spacing input rank %zu exceeds supported maximum rank %lld.",
                     self->GetViewShape().GetDimNum(), ACLNN_SPACING_MAX_SHAPE_RANK),
             return false);
    OP_CHECK(out->GetViewShape().GetDimNum() <= ACLNN_SPACING_MAX_SHAPE_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Spacing output rank %zu exceeds supported maximum rank %lld.",
                     out->GetViewShape().GetDimNum(), ACLNN_SPACING_MAX_SHAPE_RANK),
             return false);

    const op::Shape& inputShape = self->GetViewShape();
    const op::Shape& outputShape = out->GetViewShape();

    OP_CHECK(inputShape.GetDimNum() == outputShape.GetDimNum(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Spacing input/output rank mismatch: input=%zu, output=%zu.",
                     inputShape.GetDimNum(), outputShape.GetDimNum()),
             return false);

    for (size_t i = 0; i < inputShape.GetDimNum(); ++i) {
        OP_CHECK(inputShape.GetDim(i) == outputShape.GetDim(i),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Spacing input/output shape mismatch at dim %zu: input=%lld, output=%lld.",
                         i, inputShape.GetDim(i), outputShape.GetDim(i)),
                 return false);
    }
    return true;
}

aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");

    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: input_dtype=%d, output_dtype=%d",
               static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));

    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: input_format=%d, output_format=%d",
               static_cast<int>(self->GetStorageFormat()), static_cast<int>(out->GetStorageFormat()));

    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: input_rank=%zu, output_rank=%zu",
               self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());

    return ACLNN_SUCCESS;
}

} // namespace

/**
 * @brief First-stage ACLNN API: validate parameters and construct the executor.
 *
 * The returned executor owns the queued Contiguous/Spacing/ViewCopy operations.
 * No stream synchronization is performed here; execution remains asynchronous
 * until the caller synchronizes the stream after aclnnSpacing().
 */
extern "C" aclnnStatus aclnnSpacingGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSpacing, DFX_IN(self), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclnnStatus checkStatus = CheckParams(self, out);
    CHECK_RET(checkStatus == ACLNN_SUCCESS, checkStatus);

    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* spacingResult = l0op::Spacing(selfContiguous, uniqueExecutor.get());
    CHECK_RET(spacingResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* viewCopyResult = l0op::ViewCopy(spacingResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief Second-stage ACLNN API: execute the queued Spacing computation.
 *
 * CommonOpExecutorRun validates the workspace/executor relationship and submits
 * work on the supplied ACL stream. The function returns an aclnnStatus and does
 * not hide stream failures by falling back to CPU execution.
 */
extern "C" aclnnStatus aclnnSpacing(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSpacing);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}