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
 * \file argmax_tiling.cpp
 * \brief Host tiling for Argmax.
 *
 * Supported qualification profile:
 *   input  : float32, C-contiguous, rank >= 1, reduce axis is the last axis
 *   output : int64, shape equal to input shape with the last axis removed
 *
 * The kernel-facing tiling schema is defined only in
 * op_kernel/arch22/argmax_tiling_data.h. This file assigns only the
 * fields declared there. The observed target profile is [1024, 1024] with
 * axis=1, but tiling is expressed in flattened outer/inner form so rank-2
 * last-axis argmax remains the primary supported path.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/argmax_tiling_data.h"
#include "../../op_kernel/arch22/argmax_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

namespace {
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1U;
constexpr int64_t FLOAT32_SIZE = 4;
constexpr int64_t ARGMAX_BUFFER_COUNT = 2;
constexpr uint32_t ARGMAX_TILING_KEY_FALLBACK = 0U;
} // namespace

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        *coreNum <= 0, OP_LOGE(context, "Argmax: invalid AIV core number %ld", *coreNum),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(
        *ubSize == 0U, OP_LOGE(context, "Argmax: invalid UB size"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t* totalRows, int64_t* totalCols)
{
    const gert::StorageShape* inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();

    const size_t rank = inputShape.GetDimNum();
    OP_CHECK_IF(
        rank == 0U, OP_LOGE(context, "Argmax: scalar input is not supported"),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(
        axisPtr == nullptr, OP_LOGE(context, "Argmax: axis attribute is required"),
        return ge::GRAPH_FAILED);

    int64_t axis = *axisPtr;
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    OP_CHECK_IF(
        axis < 0 || axis != static_cast<int64_t>(rank) - 1,
        OP_LOGE(
            context,
            "Argmax: unsupported axis %ld for rank %zu; only the last axis is qualified",
            *axisPtr, rank),
        return ge::GRAPH_FAILED);

    int64_t outer = 1;
    constexpr int64_t maxInt64 = std::numeric_limits<int64_t>::max();
    for (size_t i = 0U; i + 1U < rank; ++i) {
        const int64_t dim = inputShape.GetDim(i);
        OP_CHECK_IF(
            dim < 0, OP_LOGE(context, "Argmax: dynamic input dimension %zu is unresolved", i),
            return ge::GRAPH_FAILED);
        if (dim == 0) {
            outer = 0;
            continue;
        }
        OP_CHECK_IF(
            outer > maxInt64 / dim,
            OP_LOGE(context, "Argmax: outer shape product overflows int64"),
            return ge::GRAPH_FAILED);
        outer *= dim;
    }

    const int64_t inner = inputShape.GetDim(rank - 1U);
    OP_CHECK_IF(
        inner < 0, OP_LOGE(context, "Argmax: dynamic reduction dimension is unresolved"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inner == 0, OP_LOGE(context, "Argmax: cannot reduce over an empty last axis"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* outputShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);
    const gert::Shape& outputShape = outputShapePtr->GetStorageShape();
    OP_CHECK_IF(
        outputShape.GetShapeSize() != outer,
        OP_LOGE(
            context, "Argmax: output size %ld does not match outer size %ld",
            outputShape.GetShapeSize(), outer),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_IF(
        inputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(context, "Argmax: only float32 input is supported in this candidate"),
        return ge::GRAPH_FAILED);

    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(
        outputDesc->GetDataType() != ge::DT_INT64,
        OP_LOGE(context, "Argmax: output must be int64"),
        return ge::GRAPH_FAILED);

    *totalRows = outer;
    *totalCols = inner;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(WS_SYS_SIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ArgmaxTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Argmax: GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    int64_t totalRows = 0;
    int64_t totalCols = 0;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalRows, &totalCols) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Argmax: GetShapeAttrsInfo failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Argmax: GetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    ArgmaxTilingData* tiling = context->GetTilingData<ArgmaxTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    std::memset(tiling, 0, sizeof(ArgmaxTilingData));

    tiling->rank = 2;
    tiling->axis = 1;
    tiling->outerSize = totalRows;
    tiling->reduceSize = totalCols;
    tiling->innerSize = 1;
    tiling->totalOut = totalRows;

    if (totalRows == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = totalCols;
        tiling->usedCoreNum = 1;
        context->SetBlockDim(1U);
        context->SetTilingKey(ARGMAX_TILING_KEY_FALLBACK);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t rowsPerCore = CeilDiv(totalRows, coreNum);
    OP_CHECK_IF(
        rowsPerCore <= 0, OP_LOGE(context, "Argmax: invalid rowsPerCore"), return ge::GRAPH_FAILED);
    const int64_t usedCoreNum = std::min(coreNum, CeilDiv(totalRows, rowsPerCore));
    OP_CHECK_IF(
        usedCoreNum <= 0, OP_LOGE(context, "Argmax: invalid usedCoreNum"), return ge::GRAPH_FAILED);

    tiling->blockFactor = rowsPerCore;

    const int64_t ubBlockSize = GetUbBlockSize(context);
    const int64_t elemAlign = std::max<int64_t>(1, ubBlockSize / FLOAT32_SIZE);
    int64_t maxColsByUb = static_cast<int64_t>(ubSize) / FLOAT32_SIZE / ARGMAX_BUFFER_COUNT;
    maxColsByUb = FloorAlign(maxColsByUb, elemAlign);
    if (maxColsByUb <= 0) {
        maxColsByUb = elemAlign;
    }

    int64_t colsPerTile = std::min(totalCols, maxColsByUb);
    if (colsPerTile <= 0) {
        colsPerTile = 1;
    }
    tiling->ubFactor = colsPerTile;
    tiling->usedCoreNum = usedCoreNum;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(ARGMAX_TILING_KEY_FALLBACK);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForArgmax([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ArgmaxCompileInfo {};

IMPL_OP_OPTILING(Argmax)
    .Tiling(ArgmaxTilingFunc)
    .TilingParse<ArgmaxCompileInfo>(TilingParseForArgmax);

} // namespace optiling