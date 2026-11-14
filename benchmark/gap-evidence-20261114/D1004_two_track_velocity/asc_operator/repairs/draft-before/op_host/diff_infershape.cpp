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
 * \file diff_infershape.cpp
 * \brief Shape inference for Diff (numpy.diff with n=1).
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {

static ge::graphStatus InferShape4Diff(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t axis = -1;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(0);
        if (axisPtr != nullptr) {
            axis = *axisPtr;
        }
    }

    const int64_t rank = static_cast<int64_t>(inputShape->GetDimNum());
    if (rank < 1) {
        return ge::GRAPH_FAILED;
    }

    int64_t normalizedAxis = axis;
    if (normalizedAxis < 0) {
        normalizedAxis += rank;
    }
    if (normalizedAxis < 0 || normalizedAxis >= rank) {
        return ge::GRAPH_FAILED;
    }

    *outputShape = *inputShape;

    const int64_t inputDim = inputShape->GetDim(static_cast<size_t>(normalizedAxis));
    int64_t outputDim = inputDim;
    if (inputDim >= 0) {
        outputDim = (inputDim > 1) ? (inputDim - 1) : 0;
    }
    outputShape->SetDim(static_cast<size_t>(normalizedAxis), outputDim);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Diff).InferShape(InferShape4Diff);

} // namespace ops