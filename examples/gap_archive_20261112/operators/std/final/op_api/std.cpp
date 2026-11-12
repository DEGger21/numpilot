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
 * @file std.cpp
 * @brief ACLNN L0 API implementation for the Std operator.
 *
 * Bounded draft support:
 * - input dtype: float32
 * - input rank: 2
 * - reduction axis: 1 (or -1, normalized to 1)
 * - output: rank-1 float32 tensor with shape [input.shape[0]]
 *
 * Unsupported dtypes, ranks, axes, empty dimensions, and non-Ascend910B
 * execution are rejected explicitly. No host fallback or implicit cast is
 * performed in this implementation.
 */

#include "std.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(NumpyStd);

static bool IsDtypeSupported(const aclTensor* x)
{
    return x != nullptr && x->GetDataType() == DataType::DT_FLOAT;
}

static bool IsAiCoreSupport(const aclTensor* x)
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2201) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Std is only supported on Ascend910B3 (DAV_2201). Current npuArch=%d.",
                static_cast<int>(npuArch));
        return false;
    }

    if (!IsDtypeSupported(x)) {
        const int dtype = (x == nullptr) ? -1 : static_cast<int>(x->GetDataType());
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Std only supports float32 input in this candidate. Current dtype=%d.",
                dtype);
        return false;
    }

    return true;
}

static bool NormalizeAndCheckAxis(const aclTensor* x, int64_t axis, int64_t& normalizedAxis)
{
    const int64_t rank = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    if (rank != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Std only supports rank-2 input in this candidate. Current rank=%lld.",
                static_cast<long long>(rank));
        return false;
    }

    normalizedAxis = axis;
    if (normalizedAxis < 0) {
        normalizedAxis += rank;
    }

    if (normalizedAxis != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Std only supports axis=1 for rank-2 input in this candidate. Current axis=%lld.",
                static_cast<long long>(axis));
        return false;
    }

    return true;
}

static bool StdInferShape(const op::Shape& xShape, int64_t normalizedAxis, op::Shape& outShape)
{
    if (normalizedAxis != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Std shape inference only supports normalized axis=1. Current axis=%lld.",
                static_cast<long long>(normalizedAxis));
        return false;
    }

    if (xShape.GetDimNum() != 2U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Std shape inference requires rank-2 input. Current rank=%zu.",
                xShape.GetDimNum());
        return false;
    }

    const int64_t rows = static_cast<int64_t>(xShape.GetDim(0));
    const int64_t cols = static_cast<int64_t>(xShape.GetDim(1));
    if (rows <= 0 || cols <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Std does not support empty input dimensions in this candidate. Shape=[%lld,%lld].",
                static_cast<long long>(rows), static_cast<long long>(cols));
        return false;
    }

    outShape.SetDimNum(1);
    outShape.SetDim(0, xShape.GetDim(0));
    return true;
}

static const aclTensor* StdAiCore(
    const aclTensor* x,
    const aclTensor* out,
    int64_t axis,
    aclOpExecutor* executor)
{
    L0_DFX(StdAiCore, x, out, axis);

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        NumpyStd,
        OP_INPUT(x),
        OP_OUTPUT(out),
        OP_ATTR(axis));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "StdAiCore launch failed. ret=%d.", static_cast<int>(ret)),
        return nullptr);
    return out;
}

const aclTensor* Std(const aclTensor* x, int64_t axis, aclOpExecutor* executor)
{
    OP_CHECK(x != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Std input x is null."), return nullptr);
    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Std executor is null."),
             return nullptr);

    if (!IsAiCoreSupport(x)) {
        return nullptr;
    }

    int64_t normalizedAxis = axis;
    if (!NormalizeAndCheckAxis(x, axis, normalizedAxis)) {
        return nullptr;
    }

    op::Shape outShape;
    if (!StdInferShape(x->GetViewShape(), normalizedAxis, outShape)) {
        return nullptr;
    }

    const aclTensor* out = executor->AllocTensor(outShape, DataType::DT_FLOAT);
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Std failed to allocate the output tensor."),
             return nullptr);

    return StdAiCore(x, out, normalizedAxis, executor);
}

} // namespace l0op