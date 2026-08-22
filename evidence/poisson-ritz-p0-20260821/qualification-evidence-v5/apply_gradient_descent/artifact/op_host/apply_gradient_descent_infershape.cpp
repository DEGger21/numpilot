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
 * \file apply_gradient_descent_infershape.cpp
 * \brief ApplyGradientDescent 算子形状推导实现
 *        语义: var_out = var - alpha * delta (非原地 NumPy oracle)
 *        输入: var (rank>=1, float32), alpha (scalar float32), delta (同 var shape)
 *        输出: var_out, shape 与 var 一致
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4ApplyGradientDescent(gert::InferShapeContext* context)
{
    // 输入 0: var, 输入 1: alpha (标量), 输入 2: delta
    const gert::Shape* varShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);

    const gert::Shape* deltaShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, deltaShape);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // 形状规则: rank>=1（不允许标量 var）
    OP_CHECK_IF(
        varShape->GetDimNum() == 0, OP_LOGE(context, "ApplyGradientDescent: var must have rank >= 1"),
        return ge::GRAPH_FAILED);

    // var 与 delta 逐元素广播一致：本算子要求两者 shape 完全一致
    OP_CHECK_IF(
        varShape->GetDimNum() != deltaShape->GetDimNum() ||
            varShape->GetShapeSize() != deltaShape->GetShapeSize(),
        OP_LOGE(
            context,
            "ApplyGradientDescent: var and delta shape mismatch: varSize=%ld, deltaSize=%ld",
            varShape->GetShapeSize(), deltaShape->GetShapeSize()),
        return ge::GRAPH_FAILED);

    // 输出 shape = var shape
    *outputShape = *varShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyGradientDescent).InferShape(InferShape4ApplyGradientDescent);

} // namespace ops