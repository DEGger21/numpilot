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
 * @file tril.cpp
 * @brief ACLNN L0 API implementation for Tril.
 *
 * Public mapping for this operator:
 *   asnumpy.tril
 *     -> Python wrapper in the isolated AsNumpy checkout
 *     -> pybind entry
 *     -> aclnnTrilGetWorkspaceSize / aclnnTril (ACLNN L2)
 *     -> l0op::Tril (this file)
 *     -> Tril Ascend C kernel
 *
 * The bounded supported profile for this candidate is:
 *   - input dtype: float32
 *   - output dtype: float32
 *   - input rank: exactly 2
 *   - layout: ND
 *   - NumPy tril semantics with the default diagonal k = 0
 *
 * Other dtypes, ranks, or unsupported platform paths are rejected here
 * instead of being silently converted or executed through a CPU fallback.
 */

#include "tril.h"

#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Tril);

namespace {

constexpr size_t kTrilRequiredRank = 2;

bool TrilInferShape(const Shape& selfShape, Shape& outShape)
{
    OP_CHECK(selfShape.GetDimNum() == kTrilRequiredRank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Tril supports only rank-2 inputs, got rank=%zu.",
                     selfShape.GetDimNum()),
             return false);

    outShape = selfShape;
    return true;
}

bool IsAiCoreSupport(const aclTensor* self)
{
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Tril is supported only on Ascend910B3 (DAV_2201), got npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);

    OP_CHECK(self->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Tril supports only float32 input, got dtype=%d.",
                     static_cast<int>(self->GetDataType())),
             return false);

    const auto selfShape = self->GetViewShape();
    OP_CHECK(selfShape.GetDimNum() == kTrilRequiredRank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Tril supports only rank-2 inputs, got rank=%zu.",
                     selfShape.GetDimNum()),
             return false);

    OP_CHECK(!self->IsEmpty(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Tril empty tensors must be handled before the L0 kernel launch."),
             return false);

    return true;
}

const aclTensor* TrilAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(TrilAiCore, self, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Tril,
                                           OP_INPUT(self),
                                           OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TrilAiCore failed to launch the Ascend C kernel."),
             return nullptr);

    return out;
}

} // namespace

const aclTensor* Tril(const aclTensor* self, aclOpExecutor* executor)
{
    OP_CHECK(self != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Tril input self must not be null."),
             return nullptr);
    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Tril executor must not be null."),
             return nullptr);

    Shape outShape;
    OP_CHECK(TrilInferShape(self->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tril infer shape failed."),
             return nullptr);

    OP_CHECK(IsAiCoreSupport(self),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tril AiCore support check failed."),
             return nullptr);

    const aclTensor* out = executor->AllocTensor(outShape, self->GetDataType());
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Tril failed to allocate the output tensor."),
             return nullptr);

    return TrilAiCore(self, out, executor);
}

} // namespace l0op