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
 * @file diff.cpp
 * @brief ACLNN L0 API implementation for Diff.
 *
 * Supported scope for this unverified candidate:
 * - input dtype: float32
 * - input rank: 2
 * - input layout: contiguous tensor already prepared by the L2 API
 * - output dtype: float32
 * - operation: first-order diff along one axis, matching numpy.diff(x, axis=axis)
 *
 * The publicly qualified scope remains limited to measured profiles. Other
 * dtypes, ranks, and invalid axes are rejected explicitly here; there is no
 * implicit cast or CPU fallback.
 */

#include "diff.h"

#include <cstddef>
#include <cstdint>

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Diff);

namespace {

constexpr int64_t kDiffOrder = 1;
constexpr size_t kSupportedRank = 2;

bool IsDtypeSupported(DataType dtype)
{
    return dtype == DataType::DT_FLOAT;
}

bool IsAiCoreSupport(const aclTensor* x)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Diff is currently supported only on Ascend910B3 (DAV_2201), got npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    OP_CHECK(IsDtypeSupported(x->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Diff currently supports only float32 input, got dtype=%d.",
                     static_cast<int>(x->GetDataType())),
             return false);

    return true;
}

bool NormalizeAxis(int64_t axis, int64_t rank, int64_t& normalizedAxis)
{
    if (rank <= 0) {
        return false;
    }
    if (axis < -rank || axis >= rank) {
        return false;
    }
    normalizedAxis = axis < 0 ? axis + rank : axis;
    return true;
}

bool DiffInferShape(const op::Shape& xShape, int64_t normalizedAxis, op::Shape& outShape)
{
    if (xShape.GetDimNum() != kSupportedRank) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Diff currently supports only rank-2 input, got rank=%zu.",
                xShape.GetDimNum());
        return false;
    }

    const int64_t axisLen = xShape.GetDim(static_cast<size_t>(normalizedAxis));
    if (axisLen < kDiffOrder) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Diff axis dimension must be at least 1, got axis=%lld, dim=%lld.",
                static_cast<long long>(normalizedAxis), static_cast<long long>(axisLen));
        return false;
    }

    outShape = xShape;
    outShape.SetDim(static_cast<size_t>(normalizedAxis), axisLen - kDiffOrder);
    return true;
}

const aclTensor* DiffAiCore(const aclTensor* x, const aclTensor* out, int64_t axis,
                            aclOpExecutor* executor)
{
    L0_DFX(DiffAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Diff,
        OP_INPUT(x), OP_OUTPUT(out), OP_ATTR(axis));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DiffAiCore launch failed, ret=%d.",
                     static_cast<int>(ret)),
             return nullptr);
    return out;
}

} // namespace

/**
 * @brief L0 API entry for Diff.
 *
 * The L2 ACLNN API is responsible for null checks, dtype checks common to all
 * operators, contiguous conversion, and final ViewCopy. This L0 implementation
 * validates the bounded support scope, infers the reduced shape, allocates the
 * output tensor, and dispatches the AI Core kernel with the normalized axis.
 *
 * @param x [in] input float32 tensor, rank 2
 * @param axis [in] diff axis, normalized to [0, rank)
 * @param executor [in] ACLNN op executor
 * @return output tensor on success, nullptr on failure
 */
const aclTensor* Diff(const aclTensor* x, int64_t axis, aclOpExecutor* executor)
{
    OP_CHECK(x != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Diff input x is null."),
             return nullptr);
    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Diff executor is null."),
             return nullptr);

    const auto& xShape = x->GetViewShape();
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());

    OP_CHECK(rank == static_cast<int64_t>(kSupportedRank),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Diff currently supports only rank-2 input, got rank=%lld.",
                     static_cast<long long>(rank)),
             return nullptr);

    int64_t normalizedAxis = 0;
    OP_CHECK(NormalizeAxis(axis, rank, normalizedAxis),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Diff axis is out of range: axis=%lld, rank=%lld.",
                     static_cast<long long>(axis), static_cast<long long>(rank)),
             return nullptr);

    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Diff IsAiCoreSupport check failed."),
             return nullptr);

    op::Shape outShape;
    OP_CHECK(DiffInferShape(xShape, normalizedAxis, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Diff infer shape failed."),
             return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, x->GetDataType());
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Diff output tensor allocation failed."),
             return nullptr);

    return DiffAiCore(x, out, normalizedAxis, executor);
}

} // namespace l0op