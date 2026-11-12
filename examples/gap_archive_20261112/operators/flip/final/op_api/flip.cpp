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
 * @file flip.cpp
 * @brief ACLNN L0 API implementation for Flip.
 *
 * This layer is invoked by the Flip L2 ACLNN adapter after input contiguity
 * handling. The operator is intentionally specialized for the currently
 * admitted profile:
 *   - input dtype: float32
 *   - input rank: 2
 *   - target SoC: Ascend910B3 / DAV_2201
 *
 * Negative axes are normalized here before the operator is launched so that
 * the tiling and kernel layers receive a non-negative axis in [0, rank).
 */

#include "flip.h"

#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Flip);

static constexpr int64_t FLIP_SUPPORTED_RANK = 2;

static bool CheckAndNormalizeAxis(const aclTensor* x, int64_t axis, int64_t& normalizedAxis)
{
    OP_CHECK(x != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Flip: input tensor must not be null."),
             return false);

    const auto& xShape = x->GetViewShape();
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());

    OP_CHECK(rank == FLIP_SUPPORTED_RANK,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Flip: unsupported input rank %lld; only rank 2 float32 input is supported.",
                     static_cast<long long>(rank)),
             return false);

    OP_CHECK(axis >= -rank && axis < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Flip: axis %lld is out of range for rank %lld.",
                     static_cast<long long>(axis), static_cast<long long>(rank)),
             return false);

    normalizedAxis = axis < 0 ? axis + rank : axis;
    return true;
}

static bool IsAiCoreSupport(const aclTensor* x)
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Flip is not supported on the current platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    OP_CHECK(x->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Flip: unsupported input dtype %d; only DT_FLOAT is admitted.",
                     static_cast<int>(x->GetDataType())),
             return false);

    return true;
}

static const aclTensor* FlipAiCore(const aclTensor* x, int64_t axis,
                                   const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(FlipAiCore, x, out);

    const auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Flip, OP_INPUT(x), OP_OUTPUT(out), OP_ATTR(axis));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FlipAiCore failed to launch the kernel."),
             return nullptr);

    return out;
}

const aclTensor* Flip(const aclTensor* x, int64_t axis, aclOpExecutor* executor)
{
    int64_t normalizedAxis = 0;
    if (!CheckAndNormalizeAxis(x, axis, normalizedAxis)) {
        return nullptr;
    }

    if (!IsAiCoreSupport(x)) {
        return nullptr;
    }

    const aclTensor* out = executor->AllocTensor(x->GetViewShape(), x->GetDataType());
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Flip: failed to allocate the output tensor."),
             return nullptr);

    return FlipAiCore(x, normalizedAxis, out, executor);
}

} // namespace l0op