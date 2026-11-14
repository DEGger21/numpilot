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
 * \file repeat_tiling.cpp
 * \brief Host tiling for the bounded Repeat candidate.
 *
 * This candidate admits only the observed float32 rank-2 axis-1 profile:
 *   input  shape [D0, D1]
 *   output shape [D0, D1 * repeats]
 * with repeats > 0.  Other rank/dtype/axis profiles are rejected explicitly
 * rather than silently cast or silently delegated to the host.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/repeat_tiling_data.h"

#include <cstring>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;
constexpr int64_t TYPE_SIZE = 4;   // float32
constexpr int64_t BUFFER_NUM = 4;  // double-buffered input and output tensors

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "Repeat: coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "Repeat: ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_IF(inputDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context, "Repeat: only float32 input is supported in this bounded candidate"),
                return ge::GRAPH_FAILED);

    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(outputDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context, "Repeat: only float32 output is supported in this bounded candidate"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context,
                                    int64_t* totalOutNum,
                                    int64_t* outerSize,
                                    int64_t* axisLen,
                                    int64_t* innerSize,
                                    int64_t* repeats)
{
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();

    auto outputShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);
    const gert::Shape& outputShape = outputShapePtr->GetStorageShape();

    OP_CHECK_IF(inputShape.GetDimNum() != 2,
                OP_LOGE(context, "Repeat: only rank-2 input is supported; got rank %zu", inputShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputShape.GetDimNum() != 2,
                OP_LOGE(context, "Repeat: only rank-2 output is supported; got rank %zu", outputShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    const int64_t inDim0 = inputShape.GetDim(0);
    const int64_t inDim1 = inputShape.GetDim(1);
    const int64_t outDim0 = outputShape.GetDim(0);
    const int64_t outDim1 = outputShape.GetDim(1);

    OP_CHECK_IF(inDim0 < 0 || inDim1 < 0 || outDim0 < 0 || outDim1 < 0,
                OP_LOGE(context, "Repeat: negative dimensions are not supported"),
                return ge::GRAPH_FAILED);

    const int64_t inTotal = inDim0 * inDim1;
    const int64_t outTotal = outDim0 * outDim1;

    if (inTotal == 0) {
        OP_CHECK_IF(outTotal != 0,
                    OP_LOGE(context, "Repeat: empty input must produce empty output"),
                    return ge::GRAPH_FAILED);
        *totalOutNum = 0;
        *outerSize = inDim0;
        *axisLen = inDim1;
        *innerSize = 1;
        *repeats = 0;
        return ge::GRAPH_SUCCESS;
    }

    // Bounded candidate: only the observed axis=1 profile is admitted.  Derive
    // the repeat factor from the already-inferred output shape so that tiling
    // does not depend on an unverified attribute/order or on host tensor reads.
    OP_CHECK_IF(outDim0 != inDim0 || inDim1 == 0 || outDim1 == 0 || (outDim1 % inDim1) != 0,
                OP_LOGE(context,
                        "Repeat: unsupported shape profile in=(%ld,%ld) out=(%ld,%ld); "
                        "only rank-2 axis=1 repeat is admitted",
                        inDim0, inDim1, outDim0, outDim1),
                return ge::GRAPH_FAILED);

    const int64_t repeatsVal = outDim1 / inDim1;
    OP_CHECK_IF(repeatsVal <= 0,
                OP_LOGE(context, "Repeat: repeats must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outTotal != inTotal * repeatsVal,
                OP_LOGE(context, "Repeat: output element count is inconsistent with repeat factor"),
                return ge::GRAPH_FAILED);

    *totalOutNum = outTotal;
    *outerSize = inDim0;
    *axisLen = inDim1;
    *innerSize = 1;
    *repeats = repeatsVal;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RepeatTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Repeat: GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Repeat: dtype check failed"), return ge::GRAPH_FAILED);

    int64_t totalOutNum = 0;
    int64_t outerSize = 0;
    int64_t axisLen = 0;
    int64_t innerSize = 0;
    int64_t repeats = 0;
    OP_CHECK_IF(GetShapeInfo(context, &totalOutNum, &outerSize, &axisLen, &innerSize, &repeats) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Repeat: shape check failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Repeat: workspace setup failed"), return ge::GRAPH_FAILED);

    RepeatTilingData* tiling = context->GetTilingData<RepeatTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    std::memset(tiling, 0, sizeof(RepeatTilingData));

    if (totalOutNum == 0) {
        context->SetBlockDim(1);
        context->SetTilingKey(0);
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    OP_CHECK_IF(ubBlockSize <= 0,
                OP_LOGE(context, "Repeat: invalid UB block size"), return ge::GRAPH_FAILED);

    int64_t blockFactor = CeilAlign(CeilDiv(totalOutNum, coreNum), ubBlockSize);
    if (blockFactor < 1) {
        blockFactor = 1;
    }

    int64_t usedCoreNum = CeilDiv(totalOutNum, blockFactor);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / TYPE_SIZE, BUFFER_NUM), ubBlockSize);
    if (ubFactor < 1) {
        ubFactor = 1;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->totalNum = totalOutNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->outerSize = outerSize;
    tiling->axisLen = axisLen;
    tiling->innerSize = innerSize;
    tiling->repeats = repeats;

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRepeat([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct RepeatCompileInfo {};  // Required for graph-mode compilation.

IMPL_OP_OPTILING(Repeat).Tiling(RepeatTilingFunc).TilingParse<RepeatCompileInfo>(TilingParseForRepeat);

}  // namespace optiling