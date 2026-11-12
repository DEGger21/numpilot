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
 * \file flip_tiling.cpp
 * \brief Flip arch22 tiling implementation.
 *
 * Supported candidate scope:
 *   - dtype: float32
 *   - rank: 2
 *   - axis: 1 (NumPy last-axis equivalent); axis -1 is normalized for rank 2
 *   - layout reaching the kernel: contiguous tensors as produced by the ACLNN L2 contiguous path
 *
 * Other axes, ranks, and dtypes are rejected explicitly in tiling rather than silently
 * reinterpreted or cast.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

#include "../../op_kernel/arch22/flip_tiling_data.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

namespace {
constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1U;
constexpr int64_t FLIP_TYPE_SIZE = static_cast<int64_t>(sizeof(float));
constexpr int64_t FLIP_SUPPORTED_RANK = 2;
constexpr int64_t FLIP_SUPPORTED_AXIS = 1;
constexpr int64_t FLIP_UB_BUFFER_COUNT = 2; // one input staging buffer + one output staging buffer
constexpr uint32_t FLIP_DEFAULT_UB_BLOCK_SIZE = 32U;
constexpr uint32_t FLIP_TILING_KEY_FP32 = 0U;
constexpr int64_t FLIP_MAX_UB_FACTOR = 4096;

static inline int64_t AlignUp(int64_t value, int64_t align)
{
    if (align <= 1) {
        return value;
    }
    return CeilAlign(value, align);
}

static inline int64_t SafeMul(int64_t lhs, int64_t rhs, bool& overflow)
{
    overflow = false;
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    if (lhs > 0 && rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) {
        overflow = true;
        return 0;
    }
    return lhs * rhs;
}
} // namespace

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, ubSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, coreNum);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfoPtr);
    *coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(*coreNum <= 0, OP_LOGE(context, "Flip: aiv core num is invalid"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0U, OP_LOGE(context, "Flip: ub size is zero"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t* dim0, int64_t* dim1, int64_t* axis)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, dim0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dim1);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    const gert::StorageShape* inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();

    const gert::StorageShape* outputShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);
    const gert::Shape& outputShape = outputShapePtr->GetStorageShape();

    OP_CHECK_IF(
        inputShape.GetDimNum() != static_cast<size_t>(FLIP_SUPPORTED_RANK),
        OP_LOGE(
            context, "Flip: unsupported input rank %zu, candidate supports rank 2",
            inputShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        outputShape.GetDimNum() != static_cast<size_t>(FLIP_SUPPORTED_RANK),
        OP_LOGE(
            context, "Flip: unsupported output rank %zu, candidate supports rank 2",
            outputShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    const int64_t inDim0 = inputShape.GetDim(0);
    const int64_t inDim1 = inputShape.GetDim(1);
    OP_CHECK_IF(
        inDim0 < 0 || inDim1 < 0,
        OP_LOGE(context, "Flip: negative input dimensions are invalid: %ld x %ld", inDim0, inDim1),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        outputShape.GetDim(0) != inDim0 || outputShape.GetDim(1) != inDim1,
        OP_LOGE(
            context, "Flip: output shape must equal input shape, got output %ld x %ld for input %ld x %ld",
            outputShape.GetDim(0), outputShape.GetDim(1), inDim0, inDim1),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_IF(
        inputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(
            context, "Flip: unsupported input dtype %d, candidate supports float32 only",
            static_cast<int32_t>(inputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(
        outputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(
            context, "Flip: unsupported output dtype %d, candidate produces float32 only",
            static_cast<int32_t>(outputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(axisPtr == nullptr, OP_LOGE(context, "Flip: required attribute axis is missing"), return ge::GRAPH_FAILED);

    int64_t normalizedAxis = *axisPtr;
    if (normalizedAxis < 0) {
        normalizedAxis += FLIP_SUPPORTED_RANK;
    }
    OP_CHECK_IF(
        normalizedAxis != FLIP_SUPPORTED_AXIS,
        OP_LOGE(
            context, "Flip: unsupported axis %ld, candidate supports last-axis flip on rank-2 tensors",
            *axisPtr),
        return ge::GRAPH_FAILED);

    *dim0 = inDim0;
    *dim1 = inDim1;
    *axis = normalizedAxis;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FlipTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Flip: failed to get platform information"), return ge::GRAPH_FAILED);

    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t axis = FLIP_SUPPORTED_AXIS;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &dim0, &dim1, &axis) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Flip: failed to get shape or attribute information"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Flip: failed to set workspace size"), return ge::GRAPH_FAILED);

    FlipTilingData* tiling = context->GetTilingData<FlipTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    std::memset(tiling, 0, sizeof(FlipTilingData));

    tiling->rows = dim0;
    tiling->cols = dim1;
    tiling->axis = axis;
    tiling->flipDim0 = 0;
    tiling->flipDim1 = 1;

    bool overflow = false;
    const int64_t totalNum = SafeMul(dim0, dim1, overflow);
    OP_CHECK_IF(
        overflow,
        OP_LOGE(context, "Flip: element count overflows int64 for shape %ld x %ld", dim0, dim1),
        return ge::GRAPH_FAILED);

    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 1;
        context->SetBlockDim(1U);
        context->SetTilingKey(FLIP_TILING_KEY_FP32);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t ubBlockSize = GetUbBlockSize(context);
    if (ubBlockSize == 0U) {
        ubBlockSize = FLIP_DEFAULT_UB_BLOCK_SIZE;
    }
    int64_t alignElements = static_cast<int64_t>(ubBlockSize) / FLIP_TYPE_SIZE;
    if (alignElements <= 0) {
        alignElements = 1;
    }

    // Parallelism is rows: every output row is an independent flip, so one
    // core owns `blockFactor` consecutive rows (FlipTilingData documents
    // blockFactor as rows per core, not elements).
    const int64_t rawBlockFactor = CeilDiv(dim0, coreNum);
    const int64_t blockFactor = rawBlockFactor > 0 ? rawBlockFactor : 1;
    OP_CHECK_IF(blockFactor <= 0, OP_LOGE(context, "Flip: computed invalid block factor"), return ge::GRAPH_FAILED);

    const int64_t usedCoreNum = CeilDiv(dim0, blockFactor);
    OP_CHECK_IF(
        usedCoreNum <= 0 || usedCoreNum > coreNum,
        OP_LOGE(
            context, "Flip: invalid used core count %ld for total %ld, core num %ld",
            usedCoreNum, totalNum, coreNum),
        return ge::GRAPH_FAILED);

    const int64_t ubCapacityElements =
        static_cast<int64_t>(ubSize / static_cast<uint64_t>(FLIP_TYPE_SIZE)) / FLIP_UB_BUFFER_COUNT;
    int64_t ubFactor = FloorAlign(ubCapacityElements, alignElements);
    // Keep the UB plan bounded: two queue buffers plus the byte-valued gather
    // offsets must fit comfortably inside the 192 KiB UB of the 910B.
    ubFactor = std::min(ubFactor, FLIP_MAX_UB_FACTOR);
    if (ubFactor < alignElements) {
        ubFactor = alignElements;
    }

    const int64_t totalAligned = AlignUp(totalNum, alignElements);
    ubFactor = std::min(ubFactor, totalAligned);
    if (ubFactor <= 0) {
        ubFactor = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(FLIP_TILING_KEY_FP32);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFlip([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct FlipCompileInfo {};

IMPL_OP_OPTILING(Flip).Tiling(FlipTilingFunc).TilingParse<FlipCompileInfo>(TilingParseForFlip);

} // namespace optiling