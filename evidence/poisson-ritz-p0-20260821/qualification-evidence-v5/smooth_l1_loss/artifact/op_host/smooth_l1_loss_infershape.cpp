/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_infershape.cpp
 * \brief SmoothL1Loss 算子形状推导实现
 *
 * 语义（NumPy oracle, mean 归约, beta=1）:
 *   error = predict - label
 *   absolute = |error|
 *   bounded = min(absolute, 1.0)
 *   loss = mean(bounded^2 / 2 + (absolute - bounded))
 *
 * 输入: predict, label（float32, rank>=1, 形状一致, 连续）
 * 输出: y（float32 标量, shape = []）
 *
 * 注意：内部注册 OpType 使用全局唯一且不含数字的 NumPilotSmoothMeanLoss。
 * CANN 9.0 的构建期与运行期 CamelCase 转文件名逻辑在嵌入数字（如 L1）上
 * 不一致（561103/561108），因此内部 OpType 必须数字无关；对外 ACLNN 符号
 * 仍为 aclnnSmoothL1Loss*，Python API 仍为 asnumpy.smooth_l1_loss。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// NumPilotSmoothMeanLoss 形状推导：输出为标量（mean 归约）
static ge::graphStatus InferShape4NumPilotSmoothMeanLoss(gert::InferShapeContext* context)
{
    const gert::Shape* predictShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);

    const gert::Shape* labelShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShape);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // 形状规则：rank >= 1（不接收标量输入）
    OP_CHECK_IF(predictShape->GetDimNum() == 0,
        OP_LOGE(context, "NumPilotSmoothMeanLoss: predict rank must be >= 1, got scalar"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(labelShape->GetDimNum() == 0,
        OP_LOGE(context, "NumPilotSmoothMeanLoss: label rank must be >= 1, got scalar"),
        return ge::GRAPH_FAILED);

    // predict 与 label 形状必须一致
    OP_CHECK_IF(predictShape->GetDimNum() != labelShape->GetDimNum(),
        OP_LOGE(context, "NumPilotSmoothMeanLoss: predict rank %lu != label rank %lu",
            predictShape->GetDimNum(), labelShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < predictShape->GetDimNum(); i++) {
        OP_CHECK_IF(predictShape->GetDim(i) != labelShape->GetDim(i),
            OP_LOGE(context, "NumPilotSmoothMeanLoss: predict dim[%lu]=%ld != label dim[%lu]=%ld",
                i, predictShape->GetDim(i), i, labelShape->GetDim(i)),
            return ge::GRAPH_FAILED);
    }

    // mean 归约：输出为标量
    *outputShape = gert::Shape();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NumPilotSmoothMeanLoss).InferShape(InferShape4NumPilotSmoothMeanLoss);

} // namespace ops