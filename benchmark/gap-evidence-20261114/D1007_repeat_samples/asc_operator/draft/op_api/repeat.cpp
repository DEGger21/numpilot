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
 * @file repeat.cpp
 * @brief ACLNN L0 API implementation for Repeat.
 *
 * Bounded support for this draft:
 * - input dtype: float32
 * - input rank: 2
 * - axis: 1 or -1 (normalized to 1)
 * - repeats: positive scalar
 *
 * The public NumPy-compatible contract is numpy.repeat(a, repeats, axis=1)
 * for the observed CPU profile. Out-of-scope dtype/shape/axis/repeats cases
 * are rejected here rather than silently cast or executed on the host.
 */

#include "repeat.h"

#include <cstdint>
#include <limits>

#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Repeat);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool NormalizeSupportedAxis(int64_t axis, int64_t &normalizedAxis)
{
    normalizedAxis = axis;
    if (normalizedAxis == -1) {
        normalizedAxis = 1;
        return true;
    }
    return normalizedAxis == 1;
}

static bool RepeatInferShape(const op::Shape &xShape, int64_t repeats, int64_t axis, op::Shape &outShape)
{
    const size_t dimNum = xShape.GetDimNum();
    if (dimNum != 2U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Repeat only supports rank-2 input, got rank=%zu.",
                dimNum);
        return false;
    }

    int64_t normalizedAxis = 0;
    if (!NormalizeSupportedAxis(axis, normalizedAxis) || normalizedAxis != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Repeat only supports axis=1 or axis=-1 for rank-2 input, got axis=%ld.",
                static_cast<long>(axis));
        return false;
    }

    if (repeats <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Repeat requires a positive scalar repeats value, got repeats=%ld.",
                static_cast<long>(repeats));
        return false;
    }

    const int64_t inputDim1 = xShape.GetDim(1);
    if (inputDim1 < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Repeat received invalid negative input dimension, dim1=%ld.",
                static_cast<long>(inputDim1));
        return false;
    }

    if (inputDim1 > 0 && repeats > std::numeric_limits<int64_t>::max() / inputDim1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Repeat output dimension overflows int64: dim1=%ld, repeats=%ld.",
                static_cast<long>(inputDim1),
                static_cast<long>(repeats));
        return false;
    }

    outShape = xShape;
    outShape.SetDim(1, inputDim1 * repeats);
    return true;
}

static bool IsAiCoreSupport(const aclTensor *x)
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Repeat is supported only on Ascend910B3 (DAV_2201), got npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    OP_CHECK(CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Repeat supports only float32 input, got dtype=%d.",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static const aclTensor *RepeatAiCore(const aclTensor *x, const aclTensor *out, int64_t repeats, int64_t axis,
                                     aclOpExecutor *executor)
{
    L0_DFX(RepeatAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Repeat,
        OP_INPUT(x),
        OP_OUTPUT(out),
        OP_ATTR(repeats, axis));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RepeatAiCore launch failed."),
             return nullptr);
    return out;
}

const aclTensor *Repeat(const aclTensor *x, int64_t repeats, int64_t axis, aclOpExecutor *executor)
{
    OP_CHECK(x != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Repeat input tensor is null."),
             return nullptr);
    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Repeat executor is null."),
             return nullptr);

    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Repeat IsAiCoreSupport check failed."),
             return nullptr);

    op::Shape outShape;
    OP_CHECK(RepeatInferShape(x->GetViewShape(), repeats, axis, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Repeat infer shape failed."),
             return nullptr);

    int64_t normalizedAxis = 0;
    OP_CHECK(NormalizeSupportedAxis(axis, normalizedAxis),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Repeat axis normalization failed."),
             return nullptr);

    const aclTensor *out = executor->AllocTensor(outShape, x->GetDataType());
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Repeat output tensor allocation failed."),
             return nullptr);

    return RepeatAiCore(x, out, repeats, normalizedAxis, executor);
}

} // namespace l0op