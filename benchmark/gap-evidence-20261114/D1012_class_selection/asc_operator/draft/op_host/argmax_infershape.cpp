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
 * \file argmax_infershape.cpp
 * \brief Shape inference for the Argmax operator.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

namespace ops {
namespace {

constexpr size_t kInputIndex = 0U;
constexpr size_t kOutputIndex = 0U;
constexpr size_t kAxisAttrIndex = 0U;

ge::graphStatus SetArgmaxOutputShape(
    gert::InferShapeContext* context, const gert::Shape& input_shape, int64_t axis)
{
    gert::Shape* output_shape = context->GetOutputShape(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    const int64_t rank = static_cast<int64_t>(input_shape.GetDimNum());
    output_shape->SetDimNum(0U);

    if (rank == 0) {
        if (axis == 0 || axis == -1) {
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGE(context, "Argmax: axis is out of range for a scalar input.");
        return ge::GRAPH_FAILED;
    }

    if (axis < -rank || axis >= rank) {
        OP_LOGE(context, "Argmax: axis is out of range for the input rank.");
        return ge::GRAPH_FAILED;
    }
    if (axis < 0) {
        axis += rank;
    }

    for (int64_t dim = 0; dim < rank; ++dim) {
        if (dim == axis) {
            continue;
        }
        output_shape->AppendDim(input_shape.GetDim(static_cast<size_t>(dim)));
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus InferShape4Argmax(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* input_shape = context->GetInputShape(kInputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    gert::Shape* output_shape = context->GetOutputShape(kOutputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* axis_ptr = attrs->GetAttrPointer<int64_t>(kAxisAttrIndex);
    if (axis_ptr == nullptr) {
        // NumPy default: axis=None flattens the input to a scalar index.
        output_shape->SetDimNum(0U);
        return ge::GRAPH_SUCCESS;
    }

    return SetArgmaxOutputShape(context, *input_shape, *axis_ptr);
}

IMPL_OP_INFERSHAPE(Argmax).InferShape(InferShape4Argmax);

} // namespace ops