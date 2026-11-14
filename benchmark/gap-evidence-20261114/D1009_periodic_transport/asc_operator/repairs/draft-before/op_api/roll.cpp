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
 * @file roll.cpp
 * @brief ACLNN L0 API implementation for numpy.roll semantics.
 *
 * Public mapping:
 *   asnumpy.roll(a, shift, axis=...)
 *     -> Python wrapper
 *     -> pybind symbol
 *     -> native adapter
 *     -> aclnnRollGetWorkspaceSize / aclnnRoll
 *     -> l0op::Roll (this file)
 *     -> Roll kernel
 *
 * This candidate supports float32 tensors on Ascend910B3 only. Other input
 * dtypes are rejected explicitly; no implicit cast or host fallback is used.
 */

#include "roll.h"

#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Roll);

static const std::initializer_list<op::DataType> ROLL_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsRollAiCoreSupport(const aclTensor* self)
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(
        npuArch == NpuArch::DAV_2201,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Roll is supported only on Ascend910B3 (DAV_2201), current npuArch=%d.",
            static_cast<int>(npuArch)),
        return false);

    OP_CHECK(
        CheckType(self->GetDataType(), ROLL_AICORE_DTYPE_SUPPORT_LIST),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Roll supports only float32 input. Got dtype=%d.",
            static_cast<int>(self->GetDataType())),
        return false);

    return true;
}

static bool RollInferShape(const op::Shape& selfShape, op::Shape& outShape)
{
    // numpy.roll preserves the input shape and dtype.
    outShape = selfShape;
    return true;
}

static bool NormalizeRollAxis(const aclTensor* self, int64_t& axis)
{
    const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    OP_CHECK(
        rank > 0,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Roll does not support a scalar tensor with an explicit axis."),
        return false);

    if (axis < 0) {
        axis += rank;
    }

    OP_CHECK(
        axis >= 0 && axis < rank,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Roll axis out of range after normalization: axis=%ld, rank=%ld.",
            static_cast<long>(axis),
            static_cast<long>(rank)),
        return false);

    return true;
}

static const aclTensor* RollAiCore(
    const aclTensor* self,
    int64_t shift,
    int64_t axis,
    const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(RollAiCore, self, shift, axis, out);

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Roll,
        OP_INPUT(self),
        OP_OUTPUT(out),
        OP_ATTR(shift, axis));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(
            ACLNN_ERR_INNER_NULLPTR,
            "RollAiCore failed to add Roll to the launcher list."),
        return nullptr);

    return out;
}

/**
 * @brief ACLNN L0 API entry for Roll.
 *
 * @param self  float32 input tensor
 * @param shift NumPy roll shift, may be negative and larger than the axis extent
 * @param axis  scalar axis; negative axes are normalized to [0, rank)
 * @param executor ACLNN executor
 * @return newly allocated output tensor, or nullptr on validation/allocation error
 */
const aclTensor* Roll(
    const aclTensor* self,
    int64_t shift,
    int64_t axis,
    aclOpExecutor* executor)
{
    OP_CHECK(
        self != nullptr && executor != nullptr,
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Roll received a null tensor or executor."),
        return nullptr);

    OP_CHECK(
        IsRollAiCoreSupport(self),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Roll IsRollAiCoreSupport check failed."),
        return nullptr);

    OP_CHECK(
        NormalizeRollAxis(self, axis),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Roll axis validation failed."),
        return nullptr);

    op::Shape outShape;
    OP_CHECK(
        RollInferShape(self->GetViewShape(), outShape),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Roll infer shape failed."),
        return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, self->GetDataType());
    OP_CHECK(
        out != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Roll failed to allocate output tensor."),
        return nullptr);

    return RollAiCore(self, shift, axis, out, executor);
}

} // namespace l0op