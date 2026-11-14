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
 * @file nextafter.cpp
 * @brief ACLNN L0 API implementation for Nextafter.
 *
 * NumPy reference:
 *   numpy.nextafter(x1, x2)
 *
 * Supported candidate scope for this draft:
 *   - Ascend910B3 (DAV_2201)
 *   - float32 inputs with float32 output
 *   - two positional inputs, no keyword arguments
 *
 * Out-of-scope dtypes and platforms are rejected explicitly. This file does
 * not perform host fallback or implicit dtype conversion.
 */

#include "nextafter.h"

#include <initializer_list>

#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Nextafter);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsAiCoreSupport(const aclTensor* x1, const aclTensor* x2)
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter is supported only on Ascend910B3 (DAV_2201); got npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    OP_CHECK(CheckType(x1->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
                 CheckType(x2->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter supports only float32 inputs; got x1_dtype=%d, x2_dtype=%d.",
                     static_cast<int>(x1->GetDataType()), static_cast<int>(x2->GetDataType())),
             return false);

    OP_CHECK(x1->GetDataType() == x2->GetDataType(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter input dtypes must match; got x1_dtype=%d, x2_dtype=%d.",
                     static_cast<int>(x1->GetDataType()), static_cast<int>(x2->GetDataType())),
             return false);

    return true;
}

static bool NextafterInferShape(const op::Shape& x1Shape, const op::Shape& x2Shape, op::Shape& outShape)
{
    OP_CHECK(BroadcastInferShape(x1Shape, x2Shape, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter shape inference failed for x1_shape=%s and x2_shape=%s.",
                     op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString()),
             return false);
    return true;
}

static const aclTensor* NextafterAiCore(const aclTensor* x1, const aclTensor* x2,
                                        const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(NextafterAiCore, x1, x2, out);

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Nextafter,
                                                 OP_INPUT(x1, x2),
                                                 OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "NextafterAiCore launch failed."),
             return nullptr);
    return out;
}

/**
 * @brief L0 API entry for Nextafter.
 *
 * The L2 API is responsible for contiguous conversion and ViewCopy handling.
 * This L0 entry validates the candidate scope, infers the broadcasted output
 * shape, allocates the float32 output tensor, and launches the AI Core kernel.
 */
const aclTensor* Nextafter(const aclTensor* x1, const aclTensor* x2, aclOpExecutor* executor)
{
    OP_CHECK(x1 != nullptr && x2 != nullptr && executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Nextafter received a null input or executor."),
             return nullptr);

    OP_CHECK(IsAiCoreSupport(x1, x2),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Nextafter IsAiCoreSupport check failed."),
             return nullptr);

    Shape outShape;
    OP_CHECK(NextafterInferShape(x1->GetViewShape(), x2->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Nextafter infer shape failed."),
             return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, DataType::DT_FLOAT);
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Nextafter output allocation failed."),
             return nullptr);

    return NextafterAiCore(x1, x2, out, executor);
}

} // namespace l0op