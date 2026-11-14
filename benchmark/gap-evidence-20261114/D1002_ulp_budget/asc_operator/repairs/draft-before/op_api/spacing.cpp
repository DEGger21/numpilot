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
 * @file spacing.cpp
 * @brief ACLNN L0 API implementation for Spacing.
 *
 * Supported scope for this unverified candidate:
 * - input dtype: float32 only
 * - output dtype: float32, shape is identical to the input shape
 * - layout: contiguous tensors are prepared by the L2 API before this call
 *
 * Other input dtypes are rejected explicitly. This file performs no implicit
 * dtype conversion and contains no host/CPU fallback.
 */

#include "spacing.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Spacing);

static bool IsAiCoreSupport(const aclTensor* x)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Spacing is supported only on Ascend910B3 (DAV_2201); current npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    OP_CHECK(x->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Spacing supports only FLOAT32 (DT_FLOAT) input; got dtype=%d.",
                     static_cast<int>(x->GetDataType())),
             return false);

    return true;
}

static const aclTensor* SpacingAiCore(const aclTensor* x, const aclTensor* out,
                                      aclOpExecutor* executor)
{
    L0_DFX(SpacingAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Spacing,
        OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SpacingAiCore launch failed."),
             return nullptr);

    return out;
}

/**
 * @brief L0 API entry point.
 *
 * Semantics are delegated to the registered Spacing Ascend C kernel. The output
 * shape equals the input shape, and the output dtype is float32.
 */
const aclTensor* Spacing(const aclTensor* x, aclOpExecutor* executor)
{
    OP_CHECK(x != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Spacing input x is null."),
             return nullptr);
    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Spacing executor is null."),
             return nullptr);

    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Spacing IsAiCoreSupport check failed."),
             return nullptr);

    const op::Shape outShape = x->GetViewShape();
    const aclTensor* out = executor->AllocTensor(outShape, DataType::DT_FLOAT);
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Spacing output allocation failed."),
             return nullptr);

    if (x->IsEmpty()) {
        return out;
    }

    return SpacingAiCore(x, out, executor);
}

} // namespace l0op