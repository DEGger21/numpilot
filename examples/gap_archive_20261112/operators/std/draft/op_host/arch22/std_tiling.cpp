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
 * \file std_tiling.cpp
 * \brief Host tiling for the bounded float32 numpy.std candidate (reduce over axis = 1).
 *
 * Admitted profile (see gap-spec.json): 2-D contiguous float32 input, reduction along
 * axis = 1 (or the equivalent axis = -1 on a rank-2 tensor), float32 output of shape [rows].
 * Every input outside this bounded profile is rejected here with an explicit error instead of
 * being silently cast, reshaped, or executed on the host.
 *
 * StdTilingData contract shared with op_kernel/arch22/std_tiling_data.h:
 *   int64_t rows        - number of rows of the 2-D input
 *   int64_t cols        - reduction length (input.shape[1])
 *   int64_t rowsPerCore - rows owned by one AIV core, ceil(rows / coreNum)
 *   int64_t colTileLen  - columns moved into UB per inner iteration
 *   int64_t colTileNum  - ceil(cols / colTileLen)
 *
 * The kernel entry uses TilingKey 0 (single float32 specialization) and derives its row window
 * from GetBlockIdx() * rowsPerCore, clamping against StdTilingData::rows.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "securec.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/std_tiling_data.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

namespace {
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1U;
constexpr int64_t TYPE_SIZE = static_cast<int64_t>(sizeof(float));
// x tile (double buffered) + deviation tile (double buffered) + scalar scratch space.
constexpr int64_t UB_BUFFER_COUNT = 4;
constexpr int64_t SUPPORTED_AXIS = 1;
static_assert(sizeof(float) == 4, "Std candidate assumes 32-bit float");
} // namespace

// Fetch AIV core count and UB capacity for the pinned Ascend910B3 target.
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(
        *coreNum <= 0, OP_LOGE(context, "Std: invalid AIV core number %ld", *coreNum),
        return ge::GRAPH_FAILED);

    *ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0U, OP_LOGE(context, "Std: UB size is zero"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// The bounded candidate only reduces the innermost axis of a rank-2 tensor. The def file may
// expose `axis` as an attribute (validated here) or leave it implicit because the Python and
// ACLNN layers already restrict axis to {1, -1}; both layouts are handled without guessing a
// different reduction direction.
static ge::graphStatus GetReductionAxis(gert::TilingContext* context, int64_t* axis)
{
    *axis = SUPPORTED_AXIS;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetAttrNum() == 0U) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t* axisPtr = attrs->GetInt(0);
    OP_CHECK_IF(
        axisPtr == nullptr,
        OP_LOGE(context, "Std: attribute 0 must be the int64 axis (only axis=1 is admitted)"),
        return ge::GRAPH_FAILED);
    *axis = *axisPtr;
    return ge::GRAPH_SUCCESS;
}

// Validate the admitted dtype/shape/axis profile and extract the reduction geometry.
static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t axis, int64_t* rows, int64_t* cols)
{
    OP_CHECK_IF(
        axis != SUPPORTED_AXIS && axis != -SUPPORTED_AXIS,
        OP_LOGE(context, "Std: only axis=1 (or -1 on a 2-D input) is admitted, got axis=%ld", axis),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();
    OP_CHECK_IF(
        inputShape.GetDimNum() != 2U,
        OP_LOGE(context, "Std: only rank-2 input is admitted, got rank %zu", inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    *rows = inputShape.GetDim(0);
    *cols = inputShape.GetDim(1);
    OP_CHECK_IF(
        *rows < 0 || *cols < 0,
        OP_LOGE(context, "Std: negative input extent rows=%ld cols=%ld", *rows, *cols),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *cols == 0,
        OP_LOGE(context, "Std: reducing a zero-length axis is rejected by this candidate"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* outputShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);
    const gert::Shape& outputShape = outputShapePtr->GetStorageShape();
    OP_CHECK_IF(
        outputShape.GetDimNum() != 1U,
        OP_LOGE(context, "Std: only rank-1 output is admitted, got rank %zu", outputShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputShape.GetDim(0) != *rows,
        OP_LOGE(
            context, "Std: output length %ld does not match the %ld input rows",
            outputShape.GetDim(0), *rows),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_IF(
        inputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(
            context, "Std: only float32 input is admitted, got ge::DataType %d",
            static_cast<int32_t>(inputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(
        outputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(
            context, "Std: only float32 output is admitted, got ge::DataType %d",
            static_cast<int32_t>(outputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// A workspace slot must always be declared, even when its size is zero.
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus StdTilingFunc(gert::TilingContext* context)
{
    // 1. Platform limits of the pinned SoC.
    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Std: GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    // 2. Bounded semantics: axis, dtype, rank, and output geometry.
    int64_t axis = SUPPORTED_AXIS;
    OP_CHECK_IF(
        GetReductionAxis(context, &axis) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Std: GetReductionAxis failed"), return ge::GRAPH_FAILED);

    int64_t rows = 0;
    int64_t cols = 0;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, axis, &rows, &cols) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Std: GetShapeAttrsInfo failed"), return ge::GRAPH_FAILED);

    // 3. Workspace declaration (no extra device memory is required by this candidate).
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Std: GetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    // 4. Publish tiling data.
    StdTilingData* tiling = context->GetTilingData<StdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(StdTilingData), 0, sizeof(StdTilingData)) != EOK,
        OP_LOGE(context, "Std: failed to clear StdTilingData"), return ge::GRAPH_FAILED);

    // An empty row set produces an empty output; no kernel work is required.
    if (rows == 0) {
        context->SetBlockDim(1U);
        context->SetTilingKey(0U);
        OP_LOGI(context, "Std tiling: empty row set, blockDim=1, tilingKey=0");
        return ge::GRAPH_SUCCESS;
    }

    // Row split across AIV cores. CeilDiv keeps every row covered; the last core may be partial.
    tiling->rows = rows;
    tiling->cols = cols;
    tiling->rowsPerCore = CeilDiv(rows, coreNum);
    OP_CHECK_IF(
        tiling->rowsPerCore <= 0,
        OP_LOGE(context, "Std: invalid rowsPerCore %ld for rows=%ld coreNum=%ld",
                tiling->rowsPerCore, rows, coreNum),
        return ge::GRAPH_FAILED);
    const int64_t usedCoreNum = CeilDiv(rows, tiling->rowsPerCore);

    // Column tile inside UB. Four float buffers keep the x tile, the deviation tile, and their
    // double buffering inside UB; the 32-byte block alignment keeps DMA extents clean.
    const int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    int64_t ubBlockElems = ubBlockSize / TYPE_SIZE;
    if (ubBlockElems < 1) {
        ubBlockElems = 1;
    }
    const int64_t ubElems = static_cast<int64_t>(ubSize) / TYPE_SIZE;
    int64_t maxColTile = FloorAlign(ubElems / UB_BUFFER_COUNT, ubBlockElems);
    if (maxColTile < ubBlockElems) {
        maxColTile = ubBlockElems;
    }
    tiling->colTileLen = std::min(cols, maxColTile);
    if (tiling->colTileLen < 1) {
        tiling->colTileLen = 1;
    }
    tiling->colTileNum = CeilDiv(cols, tiling->colTileLen);

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(0U);

    OP_LOGI(
        context,
        "Std tiling: rows=%ld cols=%ld rowsPerCore=%ld colTileLen=%ld colTileNum=%ld "
        "blockDim=%ld tilingKey=0",
        tiling->rows, tiling->cols, tiling->rowsPerCore, tiling->colTileLen, tiling->colTileNum,
        usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForStd([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct StdCompileInfo {}; // Required by the optiling registration path.

// Tiling registration entry for the registry-invoke Std operator.
IMPL_OP_OPTILING(Std)
    .Tiling(StdTilingFunc)
    .TilingParse<StdCompileInfo>(TilingParseForStd);

} // namespace optiling