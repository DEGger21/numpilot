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
 * \file pad_tiling.cpp
 * \brief Pad operator tiling implementation for Ascend910B (arch22).
 *
 * This candidate is intentionally bounded to the observed semantic profile:
 *   - rank-2 float32 C-contiguous input
 *   - mode='constant'
 *   - non-negative pad widths
 *   - default constant value 0.0f
 *
 * Other dtypes, ranks, modes, negative pads, and rows that cannot fit in UB
 * are rejected explicitly; they are not silently cast or routed to a CPU
 * fallback.
 */

#include <cstdint>
#include <cstring>
#include <string>

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

#include "../../op_kernel/arch22/pad_tiling_data.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4; // float32
constexpr int64_t PAD_RANK = 2;
constexpr size_t PAD_ATTR_NUM = 4; // ((before_0, after_0), (before_1, after_1))

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        *coreNum <= 0, OP_LOGE(context, "Pad: invalid AIV core number %ld", *coreNum),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "Pad: UB size is zero"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t* inH, int64_t* inW, int64_t* outH, int64_t* outW,
    int64_t* padTop, int64_t* padLeft, float* constantValue)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    if (attrs->GetAttrNum() < 2) {
        OP_LOGE(
            context, "Pad: expected at least 2 attributes (pads, mode), got %zu",
            attrs->GetAttrNum());
        return ge::GRAPH_FAILED;
    }

    const gert::TypedContinuousVector<int64_t>* padsAttr =
        attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsAttr);
    if (padsAttr->GetSize() != PAD_ATTR_NUM) {
        OP_LOGE(
            context, "Pad: pads attribute must contain %zu elements, got %zu",
            PAD_ATTR_NUM, padsAttr->GetSize());
        return ge::GRAPH_FAILED;
    }
    const int64_t* pads = padsAttr->GetData();

    const char* mode = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, mode);
    if (std::strcmp(mode, "constant") != 0) {
        OP_LOGE(context, "Pad: only mode='constant' is supported, got '%s'", mode);
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < PAD_ATTR_NUM; ++i) {
        if (pads[i] < 0) {
            OP_LOGE(
                context, "Pad: negative pad width is not supported, pads[%zu]=%ld",
                i, pads[i]);
            return ge::GRAPH_FAILED;
        }
    }

    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();
    if (inputShape.GetDimNum() != static_cast<size_t>(PAD_RANK)) {
        OP_LOGE(
            context, "Pad: input rank must be %ld, got %zu",
            PAD_RANK, inputShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    *inH = inputShape.GetDim(0);
    *inW = inputShape.GetDim(1);
    *padTop = pads[0];
    *padLeft = pads[2];
    *outH = *inH + pads[0] + pads[1];
    *outW = *inW + pads[2] + pads[3];
    *constantValue = 0.0f;

    auto outputShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);
    const gert::Shape& outputShape = outputShapePtr->GetStorageShape();
    if (outputShape.GetDimNum() != static_cast<size_t>(PAD_RANK)) {
        OP_LOGE(
            context, "Pad: output rank must be %ld, got %zu",
            PAD_RANK, outputShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (outputShape.GetDim(0) != *outH || outputShape.GetDim(1) != *outW) {
        OP_LOGE(
            context, "Pad: output shape mismatch, expected [%ld, %ld], got [%ld, %ld]",
            *outH, *outW, outputShape.GetDim(0), outputShape.GetDim(1));
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    if (inputDesc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE(context, "Pad: only float32 input is supported by this candidate");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus PadTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Pad: GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    int64_t inH = 0;
    int64_t inW = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t padTop = 0;
    int64_t padLeft = 0;
    float constantValue = 0.0f;
    OP_CHECK_IF(
        GetShapeAttrsInfo(
            context, &inH, &inW, &outH, &outW, &padTop, &padLeft, &constantValue) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Pad: GetShapeAttrsInfo failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Pad: SetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    PadTilingData* tiling = context->GetTilingData<PadTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(PadTilingData), 0, sizeof(PadTilingData)) != EOK,
        OP_LOGE(context, "Pad: failed to clear tiling data"), return ge::GRAPH_FAILED);

    tiling->inH = inH;
    tiling->inW = inW;
    tiling->outH = outH;
    tiling->outW = outW;
    tiling->padTop = padTop;
    tiling->padLeft = padLeft;
    tiling->constantValue = constantValue;
    tiling->rowsPerCore = 0;

    if (outH == 0 || outW == 0) {
        context->SetBlockDim(1);
        context->SetTilingKey(0);
        return ge::GRAPH_SUCCESS;
    }

    // Each core owns a contiguous range of output rows. The kernel maps each
    // output row to either an input row or an all-constant row.
    int64_t rowsPerCore = CeilDiv(outH, coreNum);
    if (rowsPerCore <= 0) {
        rowsPerCore = 1;
    }
    int64_t usedCoreNum = CeilDiv(outH, rowsPerCore);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    tiling->rowsPerCore = rowsPerCore;

    // One UB iteration holds one input row and one output row. This bounded
    // candidate rejects a profile that cannot fit a single row in UB instead
    // of silently falling back or truncating.
    const int64_t bytesPerRow = (inW + outW) * TYPE_SIZE;
    OP_CHECK_IF(
        bytesPerRow > static_cast<int64_t>(ubSize),
        OP_LOGE(
            context,
            "Pad: one input row plus one output row requires %ld bytes, exceeds UB size %lu",
            bytesPerRow, ubSize),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct PadCompileInfo {}; // Required for graph-mode compilation compatibility.

IMPL_OP_OPTILING(Pad).Tiling(PadTilingFunc).TilingParse<PadCompileInfo>(TilingParseForPad);

} // namespace optiling