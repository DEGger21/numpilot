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
 * \file std_infershape.cpp
 * \brief Std operator shape inference.
 *
 * NumPy semantics: np.std(x, axis=k) removes dimension k from the output
 * shape. A missing axis reduces all dimensions and therefore produces a
 * rank-0 scalar shape. Negative axes are normalized against the input rank.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

namespace ops {

static constexpr size_t STD_AXIS_ATTR_INDEX = 0U;

static ge::graphStatus InferShape4Std(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* axis_ptr =
        (attrs != nullptr) ? attrs->GetAttrPointer<int64_t>(STD_AXIS_ATTR_INDEX) : nullptr;

    // np.std(x) reduces every dimension and returns a scalar.
    if (axis_ptr == nullptr) {
        output_shape->SetDimNum(0);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t input_rank = static_cast<int64_t>(input_shape->GetDimNum());
    OP_CHECK_IF(
        input_rank == 0,
        OP_LOGE(context, "Std: axis must not be specified for a rank-0 input"),
        return ge::GRAPH_FAILED);

    const int64_t original_axis = *axis_ptr;
    int64_t axis = original_axis;
    if (axis < 0) {
        axis += input_rank;
    }

    OP_CHECK_IF(
        axis < 0 || axis >= input_rank,
        OP_LOGE(context, "Std: axis %ld is out of range for input rank %ld", original_axis, input_rank),
        return ge::GRAPH_FAILED);

    output_shape->SetDimNum(0);
    for (int64_t dim_idx = 0; dim_idx < input_rank; ++dim_idx) {
        if (dim_idx == axis) {
            continue;
        }
        output_shape->AppendDim(input_shape->GetDim(dim_idx));
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Std).InferShape(InferShape4Std);

} // namespace ops