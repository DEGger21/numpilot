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
 * \file nextafter_infershape.cpp
 * \brief Shape inference implementation for Nextafter.
 */

#include <cstddef>

#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {

bool IsSameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    const size_t dimNum = lhs.GetDimNum();
    if (dimNum != rhs.GetDimNum()) {
        return false;
    }

    for (size_t i = 0U; i < dimNum; ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

} // namespace

// Nextafter is elementwise in the admitted profile: both inputs and the output
// have exactly the same shape. Broadcasting is not part of this bounded
// candidate and must be rejected explicitly rather than silently reinterpreted.
static ge::graphStatus InferShape4Nextafter(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);

    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (!IsSameShape(*x1Shape, *x2Shape)) {
        OP_LOGE(context, "Nextafter requires both input shapes to be identical; broadcasting is not supported.");
        return ge::GRAPH_FAILED;
    }

    *yShape = *x1Shape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Nextafter).InferShape(InferShape4Nextafter);

} // namespace ops