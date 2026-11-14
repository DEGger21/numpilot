/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_infershape.cpp
 * \brief Repeat operator shape inference.
 *
 * Semantics implemented here:
 *   - repeats is a required integer attribute and must be non-negative.
 *   - axis is an optional integer attribute. When present it follows NumPy
 *     axis normalization, including negative axis values. When absent the
 *     input is flattened before repeating, matching numpy.repeat(a, repeats).
 *   - The output rank equals the input rank when axis is present, otherwise
 *     the output rank is one.
 *   - Unknown dimensions (-1) are propagated as unknown output dimensions.
 */

#include <limits>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

namespace {
constexpr int64_t kUnknownDim = -1;

static ge::graphStatus InferFlattenShape(gert::InferShapeContext* context,
                                         const gert::Shape* xShape,
                                         gert::Shape* yShape,
                                         int64_t repeats)
{
    const int64_t inputSize = xShape->GetShapeSize();
    int64_t outputSize = kUnknownDim;
    if (inputSize >= 0) {
        if (inputSize > 0 && repeats > std::numeric_limits<int64_t>::max() / inputSize) {
            OP_LOGE(context,
                    "Repeat: flattened output size overflow, input size=%ld, repeats=%ld",
                    inputSize, repeats);
            return ge::GRAPH_FAILED;
        }
        outputSize = inputSize * repeats;
    }

    yShape->SetDimNum(1);
    yShape->SetDim(0, outputSize);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShape4Repeat(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* repeatsPtr = attrs->GetAttrPointer<int64_t>("repeats");
    OP_CHECK_NULL_WITH_CONTEXT(context, repeatsPtr);
    const int64_t repeats = *repeatsPtr;
    OP_CHECK_IF(repeats < 0,
                OP_LOGE(context, "Repeat: repeats must be non-negative, got %ld", repeats),
                return ge::GRAPH_FAILED);

    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>("axis");
    if (axisPtr == nullptr) {
        return InferFlattenShape(context, xShape, yShape, repeats);
    }

    const int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    OP_CHECK_IF(rank == 0,
                OP_LOGE(context, "Repeat: axis is not supported for a scalar input"),
                return ge::GRAPH_FAILED);

    int64_t axis = *axisPtr;
    OP_CHECK_IF(axis < -rank || axis >= rank,
                OP_LOGE(context, "Repeat: axis %ld out of range for rank %ld", axis, rank),
                return ge::GRAPH_FAILED);
    if (axis < 0) {
        axis += rank;
    }

    const int64_t inputDim = xShape->GetDim(static_cast<size_t>(axis));
    int64_t outputDim = kUnknownDim;
    if (inputDim >= 0) {
        if (inputDim > 0 && repeats > std::numeric_limits<int64_t>::max() / inputDim) {
            OP_LOGE(context,
                    "Repeat: output dimension overflow, input dim=%ld, repeats=%ld",
                    inputDim, repeats);
            return ge::GRAPH_FAILED;
        }
        outputDim = inputDim * repeats;
    }

    *yShape = *xShape;
    yShape->SetDim(static_cast<size_t>(axis), outputDim);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Repeat).InferShape(InferShape4Repeat);

} // namespace ops