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
 * @file argmax.cpp
 * @brief ACLNN L0 API implementation for Argmax.
 *
 * Supported semantic scope for this generated candidate:
 * - input dtype: float32 only
 * - input rank: 1..8
 * - axis: required, normalized against the input rank
 * - output dtype: int64, matching numpy.argmax observed output
 *
 * Unsupported dtypes, invalid axes, empty tensors, and private formats are
 * rejected explicitly. The L2 API is responsible for contiguous conversion and
 * for mapping the public asnumpy.argmax signature to this L0 call.
 */

#include "argmax.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Argmax);

namespace {

constexpr int64_t kArgmaxMaxShapeRank = 8;

const std::initializer_list<op::DataType> kArgmaxAiCoreDtypeSupportList = {
    DataType::DT_FLOAT
};

bool ArgmaxCheckPlatform()
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax is not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    return true;
}

bool ArgmaxCheckInput(const aclTensor* x)
{
    if (x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Argmax input is null.");
        return false;
    }

    OP_CHECK(CheckType(x->GetDataType(), kArgmaxAiCoreDtypeSupportList),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax only supports float32 input in this candidate, got dtype=%d.",
                     static_cast<int>(x->GetDataType())),
             return false);

    OP_CHECK(!IsPrivateFormat(x->GetStorageFormat()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax does not support private storage format: format=%d.",
                     static_cast<int>(x->GetStorageFormat())),
             return false);

    const int64_t rank = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    OP_CHECK(rank >= 1 && rank <= kArgmaxMaxShapeRank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax input rank must be in [1, %lld], got rank=%lld.",
                     static_cast<long long>(kArgmaxMaxShapeRank),
                     static_cast<long long>(rank)),
             return false);

    OP_CHECK(!x->IsEmpty(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax does not support empty input tensors in this candidate."),
             return false);

    return true;
}

bool ArgmaxNormalizeAxis(int64_t axis, int64_t rank, int64_t& normalizedAxis)
{
    if (axis < 0) {
        axis += rank;
    }

    OP_CHECK(axis >= 0 && axis < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argmax axis is out of range after normalization: axis=%lld, rank=%lld.",
                     static_cast<long long>(axis),
                     static_cast<long long>(rank)),
             return false);

    normalizedAxis = axis;
    return true;
}

bool ArgmaxInferShape(const op::Shape& xShape, int64_t axis, op::Shape& outShape)
{
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());

    int64_t normalizedAxis = 0;
    OP_CHECK(ArgmaxNormalizeAxis(axis, rank, normalizedAxis),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Argmax axis validation failed."),
             return false);

    for (int64_t dim = 0; dim < rank; ++dim) {
        if (dim == normalizedAxis) {
            continue;
        }
        outShape.AppendDim(xShape.GetDim(dim));
    }

    return true;
}

const aclTensor* ArgmaxAiCore(const aclTensor* x, int64_t axis,
                              const aclTensor* out, aclOpExecutor* executor)
{
    (void)executor;
    L0_DFX(ArgmaxAiCore, x, out);

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Argmax,
        OP_INPUT(x),
        OP_OUTPUT(out),
        OP_ATTR(axis));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ArgmaxAiCore launch failed."),
             return nullptr);

    return out;
}

} // namespace

const aclTensor* Argmax(const aclTensor* x, int64_t axis, aclOpExecutor* executor)
{
    if (x == nullptr || executor == nullptr) {
        return nullptr;
    }

    OP_CHECK(ArgmaxCheckPlatform(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Argmax platform check failed."),
             return nullptr);

    OP_CHECK(ArgmaxCheckInput(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Argmax input check failed."),
             return nullptr);

    op::Shape outShape;
    OP_CHECK(ArgmaxInferShape(x->GetViewShape(), axis, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Argmax infer shape failed."),
             return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, DataType::DT_INT64);
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Argmax output allocation failed."),
             return nullptr);

    int64_t normalizedAxis = axis;
    const int64_t rank = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    OP_CHECK(ArgmaxNormalizeAxis(axis, rank, normalizedAxis),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Argmax axis normalization failed."),
             return nullptr);

    return ArgmaxAiCore(x, normalizedAxis, out, executor);
}

} // namespace l0op