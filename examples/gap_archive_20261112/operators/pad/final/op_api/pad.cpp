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
 * @file pad.cpp
 * @brief ACLNN L0 API implementation for Pad.
 *
 * Supported public contract in this candidate:
 * - float32 input tensors only;
 * - rank <= 8 for host-side shape inference;
 * - non-negative pad widths with length 2 * rank;
 * - constant mode is represented by the operator's default host behaviour;
 * - unsupported dtypes, ranks, and malformed padding arrays are rejected.
 */

#include "pad.h"

#include <cstddef>
#include <cstdint>

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Pad);

namespace {

constexpr size_t kPadMaxRank = 8;

bool IsAiCoreSupport(const aclTensor* x)
{
    OP_CHECK(x->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Pad only supports DT_FLOAT in this candidate, got dtype=%d.",
                     static_cast<int>(x->GetDataType())),
             return false);

    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Pad is only qualified for Ascend910B3/DAV_2201, got npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    return true;
}

bool CheckParams(const aclTensor* x, const aclIntArray* paddings)
{
    OP_CHECK(x != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Pad input tensor is null."),
             return false);
    OP_CHECK(paddings != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Pad paddings array is null."),
             return false);

    const auto rank = x->GetViewShape().GetDimNum();
    OP_CHECK(rank <= kPadMaxRank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Pad rank must be <= %zu, got rank=%zu.",
                     kPadMaxRank, rank),
             return false);

    OP_CHECK(paddings->Size() == 2 * rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Pad paddings length must equal 2*rank=%zu, got length=%zu.",
                     2 * rank, paddings->Size()),
             return false);
    return true;
}

bool InferShape(const aclTensor* x, const aclIntArray* paddings, op::Shape& outShape)
{
    const auto rank = x->GetViewShape().GetDimNum();
    const int64_t* padData = paddings->GetData();
    OP_CHECK(padData != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Pad paddings data is null."),
             return false);

    outShape.SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t before = padData[2 * i];
        const int64_t after = padData[2 * i + 1];
        OP_CHECK(before >= 0 && after >= 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Pad does not support negative padding: axis=%zu before=%ld after=%ld.",
                         i, static_cast<long>(before), static_cast<long>(after)),
                 return false);

        const int64_t inDim = x->GetViewShape().GetDim(i);
        OP_CHECK(before <= INT64_MAX - inDim && after <= INT64_MAX - (inDim + before),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Pad output shape overflows int64: axis=%zu input=%ld before=%ld after=%ld.",
                         i, static_cast<long>(inDim), static_cast<long>(before),
                         static_cast<long>(after)),
                 return false);

        const int64_t outDim = inDim + before + after;
        OP_CHECK(outDim >= 0,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Pad output dimension is invalid: axis=%zu output=%ld.",
                         i, static_cast<long>(outDim)),
                 return false);
        outShape.SetDim(i, outDim);
    }
    return true;
}

bool SameShape(const op::Shape& lhs, const op::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

const aclTensor* PadAiCore(const aclTensor* x, const aclIntArray* paddings,
                           const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(PadAiCore, x, out);

    // Attr 1 is the mode string read by the tiling; only constant is admitted.
    const char* modeConstant = "constant";
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Pad,
        OP_INPUT(x),
        OP_OUTPUT(out),
        OP_ATTR(paddings, modeConstant));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PadAiCore launch failed."),
             return nullptr);
    return out;
}

} // namespace

/**
 * @brief L0 entry for Pad.
 *
 * The L0 API performs mode validation, host-side shape inference, output
 * tensor allocation, platform/dtype checks, and kernel launch. The L2 API is
 * responsible for parameter validation, contiguous conversion, and ViewCopy
 * handling.
 */
const aclTensor* Pad(const aclTensor* x, const aclIntArray* paddings,
                     int64_t mode, aclOpExecutor* executor)
{
    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Pad executor is null."),
             return nullptr);
    OP_CHECK(mode == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Pad only supports mode=0 (constant) in this candidate, got mode=%ld.",
                     static_cast<long>(mode)),
             return nullptr);

    if (!CheckParams(x, paddings) || !IsAiCoreSupport(x)) {
        return nullptr;
    }

    op::Shape outShape;
    if (!InferShape(x, paddings, outShape)) {
        return nullptr;
    }

    const aclTensor* out = executor->AllocTensor(outShape, x->GetDataType());
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Pad failed to allocate output tensor."),
             return nullptr);

    return PadAiCore(x, paddings, out, executor);
}

} // namespace l0op