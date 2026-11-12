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
 * @file ediff1d.cpp
 * @brief ACLNN L0 API implementation for Ediff1d.
 *
 * Semantic contract for the admitted float32 profile:
 *   out[i] = flatten(input)[i + 1] - flatten(input)[i]
 * with output length max(flatten(input).size() - 1, 0).
 *
 * This matches numpy.ediff1d(input) for the default to_begin/to_end behavior.
 * The public signature in this project accepts one positional tensor and no
 * keyword arguments. Other input dtypes are rejected explicitly rather than
 * silently cast to float32.
 */

#include "ediff1d.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Ediff1d);

static bool IsAiCoreSupport(const aclTensor* x)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Ediff1d is only supported on Ascend910B3 (DAV_2201); current npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    OP_CHECK(x->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Ediff1d only supports float32 input; got dtype=%d.",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static bool Ediff1dInferShape(const aclTensor* x, op::Shape& outShape)
{
    const auto& inShape = x->GetViewShape();
    const int64_t inputSize = inShape.GetShapeSize();
    OP_CHECK(inputSize >= 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Ediff1d requires a static input shape; got shape size %ld.",
                     static_cast<long>(inputSize)),
             return false);

    const int64_t outputSize = inputSize > 0 ? inputSize - 1 : 0;
    outShape.SetDimNum(1);
    outShape.SetDim(0, outputSize);
    return true;
}

static const aclTensor* Ediff1dAiCore(const aclTensor* x, const aclTensor* out,
                                      aclOpExecutor* executor)
{
    L0_DFX(Ediff1dAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Ediff1d,
        OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Ediff1dAiCore failed."),
             return nullptr);
    return out;
}

const aclTensor* Ediff1d(const aclTensor* x, aclOpExecutor* executor)
{
    OP_CHECK(x != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Ediff1d input is null."),
             return nullptr);
    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Ediff1d executor is null."),
             return nullptr);

    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Ediff1d AiCore support check failed."),
             return nullptr);

    op::Shape outShape;
    OP_CHECK(Ediff1dInferShape(x, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Ediff1d infer shape failed."),
             return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, x->GetDataType());
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Ediff1d output allocation failed."),
             return nullptr);

    // For empty input, and for a single-element input, NumPy returns an empty
    // output tensor. No kernel launch is required because there are no output
    // elements to compute.
    if (outShape.GetShapeSize() == 0) {
        return out;
    }

    return Ediff1dAiCore(x, out, executor);
}

} // namespace l0op