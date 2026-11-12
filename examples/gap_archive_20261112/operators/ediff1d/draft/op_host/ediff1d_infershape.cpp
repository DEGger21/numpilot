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
 * \file ediff1d_infershape.cpp
 * \brief Ediff1d shape and data type inference.
 *
 * numpy.ediff1d with the admitted one-positional-argument signature uses
 * the default flattening behavior: the input tensor is flattened to 1-D and
 * adjacent differences are returned. The output is therefore always rank 1
 * with length max(flattened_input_size - 1, 0).
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

namespace {
constexpr int64_t kUnknownDim = -1;
constexpr size_t kEdiff1dOutputRank = 1;
} // namespace

static ge::graphStatus InferShape4Ediff1d(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // gert::Shape::GetShapeSize() returns -1 when the flattened size cannot be
    // determined at compile time (dynamic shape). For a 0-D scalar this returns
    // the scalar element count (0 or 1); both cases correctly produce an empty
    // output because n - 1 is clamped at zero.
    const int64_t input_size = input_shape->GetShapeSize();

    if (input_size < 0) {
        output_shape->SetDimNum(kEdiff1dOutputRank);
        output_shape->SetDim(0, kUnknownDim);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t output_size = (input_size > 0) ? (input_size - 1) : 0;

    output_shape->SetDimNum(kEdiff1dOutputRank);
    output_shape->SetDim(0, output_size);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4Ediff1d(gert::InferDataTypeContext* context)
{
    // For the admitted float32 profile numpy.ediff1d preserves the input dtype.
    // The output dtype follows input 0; unsupported input dtypes are rejected by
    // the host tiling validation rather than silently cast here.
    const ge::DataType input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Ediff1d)
    .InferShape(InferShape4Ediff1d)
    .InferDataType(InferDataType4Ediff1d);

} // namespace ops