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
 * \file argpartition_infershape.cpp
 * \brief Argpartition shape and dtype inference.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "op_common/log/log.h"

namespace ops {

// numpy.argpartition preserves the input shape and returns index positions in
// the same layout. In this bounded candidate only float32 input is admitted
// and the output index dtype is int64, matching the observed NumPy profile.
static ge::graphStatus InferShape4Argpartition(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

// argpartition emits integer index tensors. The observed NumPy oracle result
// is int64; unsupported input dtypes are rejected rather than silently cast.
static ge::graphStatus InferDataType4Argpartition(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType inputDataType = context->GetInputDataType(0);
    if (inputDataType != ge::DT_FLOAT) {
        OP_LOGE(
            context, "Argpartition only supports float32 input in this candidate, got %d",
            static_cast<int32_t>(inputDataType));
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(0, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Argpartition)
    .InferShape(InferShape4Argpartition)
    .InferDataType(InferDataType4Argpartition);

} // namespace ops