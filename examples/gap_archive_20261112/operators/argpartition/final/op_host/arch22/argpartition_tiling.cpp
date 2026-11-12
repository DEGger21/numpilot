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
 * \file argpartition_tiling.cpp
 * \brief Argpartition tiling implementation for Ascend910B3 (arch22).
 *
 * Supported qualification profile for this draft:
 *   - rank-2 float32 input
 *   - C-contiguous layout, enforced by the operator definition via AutoContiguous
 *   - axis == 1 or axis == -1 (normalized to the last axis)
 *   - non-negative scalar kth in [0, rowLength)
 *   - int64 output with the same shape as the input
 *
 * Other dtypes, ranks, axes, empty inputs, and unsupported kth values are
 * rejected explicitly. No host-side NumPy or CPU fallback is provided.
 */

#include <algorithm>
#include <cstring>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/argpartition_tiling_data.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr size_t WORKSPACE_NUM = 1U;
constexpr uint64_t WS_SYS_SIZE = 0U;
constexpr size_t ARGPARTITION_SUPPORTED_RANK = 2U;
constexpr int64_t ARGPARTITION_LAST_AXIS = 1;

// Resolve only the bounded axis scope admitted by this draft. For rank-2 tensors,
// NumPy's last axis is 1 and may be written explicitly as 1 or -1.
static bool NormalizeAxisForRank2(int64_t axis, int64_t* normalizedAxis)
{
    if (normalizedAxis == nullptr) {
        return false;
    }
    if (axis == ARGPARTITION_LAST_AXIS || axis == -ARGPARTITION_LAST_AXIS) {
        *normalizedAxis = ARGPARTITION_LAST_AXIS;
        return true;
    }
    return false;
}

static bool TryKthAndAxis(int64_t kth, int64_t axis, int64_t rowLength,
                          int64_t* outKth, int64_t* outAxis)
{
    if (outKth == nullptr || outAxis == nullptr) {
        return false;
    }
    if (kth < 0 || kth >= rowLength) {
        return false;
    }
    int64_t normalizedAxis = 0;
    if (!NormalizeAxisForRank2(axis, &normalizedAxis)) {
        return false;
    }
    *outKth = kth;
    *outAxis = normalizedAxis;
    return true;
}

// The op definition carries kth and axis as integer attributes. Attribute order
// is resolved defensively from the admitted values so that this host stage does
// not silently reinterpret or drop either attribute. If both orders are valid
// and distinct, the call is rejected as ambiguous instead of guessing.
static ge::graphStatus GetKthAndAxis(gert::TilingContext* context, int64_t rowLength,
                                     int64_t* kth, int64_t* axis)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, kth);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const size_t attrNum = attrs->GetAttrNum();
    OP_CHECK_IF(
        attrNum != 2U,
        OP_LOGE(context, "Argpartition: expected exactly 2 attributes (kth, axis), got %zu", attrNum),
        return ge::GRAPH_FAILED);

    const int64_t* attr0 = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* attr1 = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr1);

    int64_t kth0 = 0;
    int64_t axis0 = 0;
    int64_t kth1 = 0;
    int64_t axis1 = 0;
    const bool ok0 = TryKthAndAxis(*attr0, *attr1, rowLength, &kth0, &axis0);
    const bool ok1 = TryKthAndAxis(*attr1, *attr0, rowLength, &kth1, &axis1);

    if (ok0 && ok1) {
        OP_CHECK_IF(
            kth0 != kth1 || axis0 != axis1,
            OP_LOGE(context,
                    "Argpartition: ambiguous kth/axis attributes: attr0=%ld, attr1=%ld, rowLength=%ld",
                    *attr0, *attr1, rowLength),
            return ge::GRAPH_FAILED);
        *kth = kth0;
        *axis = axis0;
        return ge::GRAPH_SUCCESS;
    }
    if (ok0) {
        *kth = kth0;
        *axis = axis0;
        return ge::GRAPH_SUCCESS;
    }
    if (ok1) {
        *kth = kth1;
        *axis = axis1;
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(context,
            "Argpartition: unsupported kth/axis attributes: attr0=%ld, attr1=%ld, rowLength=%ld",
            *attr0, *attr1, rowLength);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t* coreNum)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, coreNum);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        *coreNum <= 0,
        OP_LOGE(context, "Argpartition: invalid AI vector core number: %ld", *coreNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeDtypeAndAttrs(gert::TilingContext* context,
                                             int64_t* totalRows, int64_t* rowLength,
                                             int64_t* kth, int64_t* axis)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, totalRows);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowLength);
    OP_CHECK_NULL_WITH_CONTEXT(context, kth);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();

    OP_CHECK_IF(
        inputShape.GetDimNum() != ARGPARTITION_SUPPORTED_RANK,
        OP_LOGE(context, "Argpartition: only rank-2 input is supported, got rank %zu",
                inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    *totalRows = inputShape.GetDim(0);
    *rowLength = inputShape.GetDim(1);
    OP_CHECK_IF(
        *totalRows <= 0 || *rowLength <= 0,
        OP_LOGE(context, "Argpartition: empty input is not supported, shape=[%ld, %ld]",
                *totalRows, *rowLength),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        *rowLength > static_cast<int64_t>(ARGPARTITION_TILING_SORT_BLOCK),
        OP_LOGE(context, "Argpartition: axis length %ld exceeds the arch22 limit %d",
                *rowLength, ARGPARTITION_TILING_SORT_BLOCK),
        return ge::GRAPH_FAILED);

    auto outputShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);
    const gert::Shape& outputShape = outputShapePtr->GetStorageShape();

    OP_CHECK_IF(
        outputShape.GetDimNum() != ARGPARTITION_SUPPORTED_RANK ||
            outputShape.GetDim(0) != *totalRows ||
            outputShape.GetDim(1) != *rowLength,
        OP_LOGE(context, "Argpartition: output shape must match input shape"),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_IF(
        inputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(context, "Argpartition: only float32 input is supported, got dtype %d",
                static_cast<int32_t>(inputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(
        outputDesc->GetDataType() != ge::DT_INT64,
        OP_LOGE(context, "Argpartition: output dtype must be int64, got dtype %d",
                static_cast<int32_t>(outputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetKthAndAxis(context, *rowLength, kth, axis) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Argpartition: failed to resolve kth/axis attributes"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ArgpartitionTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Argpartition: GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    int64_t totalRows = 0;
    int64_t rowLength = 0;
    int64_t kth = 0;
    int64_t axis = 0;
    OP_CHECK_IF(
        GetShapeDtypeAndAttrs(context, &totalRows, &rowLength, &kth, &axis) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Argpartition: GetShapeDtypeAndAttrs failed"),
        return ge::GRAPH_FAILED);

    ArgpartitionTilingData* tiling = context->GetTilingData<ArgpartitionTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    std::memset(static_cast<void*>(tiling), 0, sizeof(ArgpartitionTilingData));

    tiling->totalNum = totalRows * rowLength;
    tiling->outerNum = totalRows;
    tiling->axisLen = rowLength;
    tiling->innerNum = 1;
    tiling->sliceNum = totalRows;
    tiling->kth = kth;
    tiling->alignedAxisLen = (rowLength + 7) / 8 * 8;
    tiling->axis = static_cast<int32_t>(axis);
    tiling->rank = static_cast<int32_t>(ARGPARTITION_SUPPORTED_RANK);
    tiling->sortBlockLen = ARGPARTITION_TILING_SORT_BLOCK;
    tiling->dims[0] = totalRows;
    tiling->dims[1] = rowLength;

    // The kernel performs its row split from BlockDim and totalRows. Allocate the
    // mandatory workspace slot explicitly even though this bounded implementation
    // does not require external workspace.
    size_t* workspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = static_cast<size_t>(WS_SYS_SIZE);

    int64_t blockDim = std::min<int64_t>(coreNum, totalRows);
    if (blockDim <= 0) {
        blockDim = 1;
    }
    tiling->blockFactor = blockDim;
    tiling->ubFactor = 1;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(0U);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForArgpartition([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ArgpartitionCompileInfo {};

IMPL_OP_OPTILING(Argpartition)
    .Tiling(ArgpartitionTilingFunc)
    .TilingParse<ArgpartitionCompileInfo>(TilingParseForArgpartition);

} // namespace optiling