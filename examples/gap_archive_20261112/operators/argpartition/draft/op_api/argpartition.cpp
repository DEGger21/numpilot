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
 * @file argpartition.cpp
 * @brief ACLNN L0 API implementation for Argpartition.
 *
 * Supported bounded profile for this draft:
 *   - input self: float32, rank >= 1, C-contiguous after L2 normalization
 *   - output out: int64, same shape as self
 *   - kth: single integer, normalized into [0, self.shape[axis] - 1]
 *   - axis: single integer, normalized into [0, rank - 1]
 *
 * The L2 layer performs public parameter checks, Contiguous conversion, and
 * ViewCopy handling. This L0 layer performs the platform/dtype gate, shape
 * inference, output allocation, and AI Core launcher registration.
 */

#include "argpartition.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Argpartition);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsAiCoreSupport(const aclTensor* self)
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition is not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    OP_CHECK(CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition only supports FLOAT32 input in this draft, got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);

    return true;
}

static bool ArgpartitionInferShape(const op::Shape& selfShape,
                                   int64_t& axis,
                                   int64_t kth,
                                   op::Shape& outShape)
{
    const int64_t rank = static_cast<int64_t>(selfShape.GetDimNum());
    OP_CHECK(rank > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition does not support rank-0 input in this draft."),
             return false);

    OP_CHECK(axis >= -rank && axis < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition axis is out of range: axis=%ld, rank=%ld.",
                     static_cast<long>(axis), static_cast<long>(rank)),
             return false);

    if (axis < 0) {
        axis += rank;
    }

    const int64_t axisDim = static_cast<int64_t>(selfShape.GetDim(static_cast<size_t>(axis)));
    OP_CHECK(axisDim > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition does not support an empty partition axis: axis=%ld, dim=%ld.",
                     static_cast<long>(axis), static_cast<long>(axisDim)),
             return false);

    OP_CHECK(kth >= 0 && kth < axisDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Argpartition kth is out of range: kth=%ld, axis=%ld, dim=%ld.",
                     static_cast<long>(kth), static_cast<long>(axis), static_cast<long>(axisDim)),
             return false);

    outShape = selfShape;
    return true;
}

static const aclTensor* ArgpartitionAiCore(const aclTensor* self,
                                           const aclTensor* out,
                                           int64_t kth,
                                           int64_t axis,
                                           aclOpExecutor* executor)
{
    L0_DFX(ArgpartitionAiCore, self, out);

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Argpartition,
        OP_INPUT(self),
        OP_OUTPUT(out),
        OP_ATTR(kth, axis));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "ArgpartitionAiCore launcher registration failed, ret=%d.",
                     static_cast<int>(ret)),
             return nullptr);

    return out;
}

/**
 * @brief L0 API entry for Argpartition.
 *
 * Flow:
 *   1. Validate the input dtype and target NPU architecture.
 *   2. Normalize/check axis and kth against the inferred input shape.
 *   3. Allocate the int64 output tensor with the same shape as self.
 *   4. Register the AI Core kernel launcher.
 *
 * @param self [in] Input float32 tensor.
 * @param kth [in] Partition index along axis.
 * @param axis [in] Partition axis; negative values are normalized by rank.
 * @param executor [in] ACLNN executor.
 * @return Allocated int64 output tensor, or nullptr on failure.
 */
const aclTensor* Argpartition(const aclTensor* self,
                              int64_t kth,
                              int64_t axis,
                              aclOpExecutor* executor)
{
    OP_CHECK(self != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Argpartition self is nullptr."),
             return nullptr);
    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Argpartition executor is nullptr."),
             return nullptr);

    op::Shape outShape;
    OP_CHECK(ArgpartitionInferShape(self->GetViewShape(), axis, kth, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Argpartition infer shape failed."),
             return nullptr);

    OP_CHECK(IsAiCoreSupport(self),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Argpartition IsAiCoreSupport check failed."),
             return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, DataType::DT_INT64);
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Argpartition output allocation failed."),
             return nullptr);

    return ArgpartitionAiCore(self, out, kth, axis, executor);
}

} // namespace l0op