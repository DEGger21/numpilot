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
 * \file pad_infershape.cpp
 * \brief Shape inference for Pad.
 *
 * Bounded admitted profile for this unverified candidate:
 *   - rank-2 input tensor;
 *   - non-negative pad_width flattened as {top, bottom, left, right};
 *   - mode is restricted by the tiling/API layers to the observed constant form.
 *
 * The image shape itself is mode-independent, so this infer-shape function
 * computes the padded rank-2 shape and explicitly rejects out-of-scope
 * ranks, malformed pad_width lists, negative cropping, and arithmetic overflow.
 */

#include <cstddef>
#include <cstdint>
#include <limits>

#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kPadRank = 2U;
constexpr size_t kPadWidthCount = kPadRank * 2U;

bool SafePaddedDim(int64_t input_dim, int64_t pad_before, int64_t pad_after, int64_t& output_dim)
{
    constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();

    if (input_dim < 0) {
        return false;
    }
    if (pad_before < 0 || pad_after < 0) {
        return false;
    }
    if (pad_before > kInt64Max - pad_after) {
        return false;
    }

    const int64_t pad_sum = pad_before + pad_after;
    if (input_dim > kInt64Max - pad_sum) {
        return false;
    }

    output_dim = input_dim + pad_sum;
    return true;
}
} // namespace

static ge::graphStatus InferShape4Pad(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    if (input_shape->GetDimNum() != kPadRank) {
        OP_LOGE(context, "Pad: only rank-2 input is admitted by this bounded candidate");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // Attribute 0 is pad_width, declared as ListInt and flattened in row-major
    // order: {dim0_before, dim0_after, dim1_before, dim1_after}.
    const gert::ContinuousVector* pad_width_attr = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pad_width_attr);

    if (pad_width_attr->GetSize() != kPadWidthCount) {
        OP_LOGE(
            context, "Pad: pad_width must contain exactly %zu entries for rank-2 input, got %zu",
            kPadWidthCount, pad_width_attr->GetSize());
        return ge::GRAPH_FAILED;
    }

    const int64_t* pad_width = reinterpret_cast<const int64_t*>(pad_width_attr->GetData());
    if (pad_width == nullptr) {
        OP_LOGE(context, "Pad: pad_width data is null");
        return ge::GRAPH_FAILED;
    }

    const int64_t input_dim0 = input_shape->GetDim(0);
    const int64_t input_dim1 = input_shape->GetDim(1);

    int64_t output_dim0 = 0;
    int64_t output_dim1 = 0;
    if (!SafePaddedDim(input_dim0, pad_width[0], pad_width[1], output_dim0) ||
        !SafePaddedDim(input_dim1, pad_width[2], pad_width[3], output_dim1)) {
        OP_LOGE(
            context,
            "Pad: unsupported or overflowing shape/pad_width: input=[%ld,%ld], "
            "pad_width=[%ld,%ld,%ld,%ld]",
            static_cast<long>(input_dim0), static_cast<long>(input_dim1),
            static_cast<long>(pad_width[0]), static_cast<long>(pad_width[1]),
            static_cast<long>(pad_width[2]), static_cast<long>(pad_width[3]));
        return ge::GRAPH_FAILED;
    }

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    output_shape->SetDimNum(kPadRank);
    output_shape->SetDim(0, output_dim0);
    output_shape->SetDim(1, output_dim1);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Pad).InferShape(InferShape4Pad);

} // namespace ops