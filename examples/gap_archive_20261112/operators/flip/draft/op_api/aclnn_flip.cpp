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
 * @file aclnn_flip.cpp
 * @brief ACLNN L2 two-stage API implementation for Flip.
 *
 * Symbol mapping for this bounded candidate:
 *   public asnumpy.flip
 *     -> Python adapter / pybind flip
 *     -> aclnnFlipGetWorkspaceSize / aclnnFlip
 *     -> l0op::Flip
 *     -> Flip Ascend C kernel
 *
 * Supported scope in this stage:
 *   - float32 input and float32 output only;
 *   - rank <= 8 and C-contiguous or non-contiguous view input via Contiguous();
 *   - one normalized integer axis, or kFlipAllAxes for the NumPy axis=None case.
 *
 * Other dtypes are rejected explicitly. No implicit cast or host fallback is
 * performed here.
 */

#include "aclnn_flip.h"
#include "flip.h"

#include <cstdint>
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
constexpr size_t kAclnnMaxShapeRank = 8;
constexpr int64_t kFlipAllAxes = std::numeric_limits<int64_t>::min();

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
                     "Flip supports only float32 input, got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);
    OP_CHECK(out->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Flip supports only float32 output, got dtype=%d.",
                     static_cast<int>(out->GetDataType())),
             return false);
    OP_CHECK(out->GetDataType() == self->GetDataType(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Flip input/output dtype mismatch: input=%d, output=%d.",
                     static_cast<int>(self->GetDataType()),
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
                     "Flip does not support private formats: input=%d, output=%d.",
                     static_cast<int>(selfFormat), static_cast<int>(outFormat)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, kAclnnMaxShapeRank, return false);
    OP_CHECK_MAX_DIM(out, kAclnnMaxShapeRank, return false);

    const auto& selfShape = self->GetViewShape();
    const auto& outShape = out->GetViewShape();

    OP_CHECK(selfShape.GetDimNum() == outShape.GetDimNum(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Flip rank mismatch: input_rank=%zu, output_rank=%zu.",
                     selfShape.GetDimNum(), outShape.GetDimNum()),
             return false);

    for (size_t i = 0; i < selfShape.GetDimNum(); ++i) {
        OP_CHECK(selfShape.GetDim(i) == outShape.GetDim(i),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Flip shape mismatch at dimension %zu: input=%lld, output=%lld.",
                         i,
                         static_cast<long long>(selfShape.GetDim(i)),
                         static_cast<long long>(outShape.GetDim(i))),
                 return false);
    }
    return true;
}

static bool CheckAndNormalizeAxis(const aclTensor* self, int64_t axis, int64_t* normalizedAxis)
{
    if (normalizedAxis == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Flip normalizedAxis output pointer is null.");
        return false;
    }

    *normalizedAxis = axis;
    if (axis == kFlipAllAxes) {
        return true;
    }

    const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    OP_CHECK(axis >= -rank && axis < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Flip axis %lld is out of range for rank %lld.",
                     static_cast<long long>(axis), static_cast<long long>(rank)),
             return false);

    if (axis < 0) {
        *normalizedAxis = axis + rank;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, int64_t axis, const aclTensor* out,
                               int64_t* normalizedAxis)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: input_dtype=%d, output_dtype=%d",
               static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed: input_format=%d, output_format=%d",
               static_cast<int>(self->GetStorageFormat()),
               static_cast<int>(out->GetStorageFormat()));
    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: input_rank=%zu, output_rank=%zu",
               self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());
    CHECK_COND(CheckAndNormalizeAxis(self, axis, normalizedAxis), ACLNN_ERR_PARAM_INVALID,
               "CheckAndNormalizeAxis failed: axis=%lld",
               static_cast<long long>(axis));
    return ACLNN_SUCCESS;
}
} // namespace

/**
 * @brief First-stage ACLNN API: validate parameters, build the execution plan,
 *        and return the required workspace size.
 *
 * The input may be non-contiguous. Contiguous() creates a contiguous operand
 * for the kernel. The final L0 result is copied into the caller-provided output
 * view with ViewCopy(), so non-contiguous output strides are supported.
 */
extern "C" aclnnStatus aclnnFlipGetWorkspaceSize(
    const aclTensor* self,
    int64_t axis,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFlip, DFX_IN(self), DFX_OUT(out));

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    int64_t normalizedAxis = axis;
    const auto ret = CheckParams(self, axis, out, &normalizedAxis);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* flipResult = nullptr;
    if (axis == kFlipAllAxes) {
        const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
        const aclTensor* current = selfContiguous;
        for (int64_t dim = 0; dim < rank; ++dim) {
            current = l0op::Flip(current, dim, uniqueExecutor.get());
            CHECK_RET(current != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        flipResult = current;
    } else {
        flipResult = l0op::Flip(selfContiguous, normalizedAxis, uniqueExecutor.get());
        CHECK_RET(flipResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const auto viewCopyResult = l0op::ViewCopy(flipResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief Second-stage ACLNN API: execute the prepared Flip plan on the ACL stream.
 *
 * Execution is asynchronous with respect to the host. The caller must retain
 * the tensor/stream lifetimes and synchronize the stream before consuming the
 * output. CommonOpExecutorRun() propagates ACL/runtime errors from the launch.
 */
extern "C" aclnnStatus aclnnFlip(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlip);

    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize == 0 || workspace != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}