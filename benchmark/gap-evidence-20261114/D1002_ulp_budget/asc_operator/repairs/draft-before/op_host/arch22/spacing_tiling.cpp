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
 * \file spacing_tiling.cpp
 * \brief Spacing tiling implementation for Ascend910B3 (arch22).
 *
 * This draft is specialized for the observed float32, C-contiguous profile.
 * Other input dtypes are rejected explicitly. Shape acceptance is limited to
 * element-wise tensors whose output shape matches the input shape; qualification
 * still requires the exact measured profile and device numerical validation.
 */

#include <cstring>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/spacing_tiling_data.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1U;
constexpr int64_t TYPE_SIZE = 4; // float32
// One input and one output, conservatively budgeted as double-buffered tensors.
constexpr int64_t BUFFER_TENSOR_COUNT = 4;
constexpr int64_t MIN_UB_FACTOR = 1;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, ubSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, coreNum);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum <= 0, OP_LOGE(context, "Spacing: coreNum must be positive"), return ge::GRAPH_FAILED);

    *ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0U, OP_LOGE(context, "Spacing: ubSize must be positive"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t* totalIdx, ge::DataType* dataType)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, totalIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataType);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    const int64_t inputShapeSize = inputShape->GetStorageShape().GetShapeSize();
    const int64_t outputShapeSize = outputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(
        inputShapeSize != outputShapeSize,
        OP_LOGE(
            context, "Spacing: input/output element count mismatch: input=%ld, output=%ld",
            inputShapeSize, outputShapeSize),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        *dataType != ge::DT_FLOAT,
        OP_LOGE(context, "Spacing: only float32 input is supported, got dtype=%d", static_cast<int32_t>(*dataType)),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(
        outputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(
            context, "Spacing: only float32 output is supported, got dtype=%d",
            static_cast<int32_t>(outputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    *totalIdx = inputShapeSize;
    return ge::GRAPH_SUCCESS;
}

// Even when workspace size is zero, the workspace slot must be declared so the
// runtime provides a valid workspace pointer to the kernel entry.
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SpacingTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Spacing: GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    int64_t totalIdx = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalIdx, &dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Spacing: GetShapeAttrsInfo failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Spacing: GetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    SpacingTilingData* tiling = context->GetTilingData<SpacingTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    std::memset(tiling, 0, sizeof(SpacingTilingData));

    // No template specialization is required for this float32-only draft.
    // The single registered tiling key selects the default kernel instantiation.
    context->SetTilingKey(0U);

    if (totalIdx == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1U);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(totalIdx < 0, OP_LOGE(context, "Spacing: total element count must be non-negative"),
                return ge::GRAPH_FAILED);

    const int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context, "Spacing: ubBlockSize must be positive"),
                return ge::GRAPH_FAILED);

    tiling->totalNum = totalIdx;

    // Multi-core split: align each core's element count to the DMA block size.
    const int64_t blockFactorRaw = CeilDiv(totalIdx, coreNum);
    tiling->blockFactor = CeilAlign(blockFactorRaw, ubBlockSize);
    OP_CHECK_IF(tiling->blockFactor <= 0, OP_LOGE(context, "Spacing: blockFactor must be positive"),
                return ge::GRAPH_FAILED);

    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);
    usedCoreNum = (usedCoreNum < 1) ? 1 : usedCoreNum;
    usedCoreNum = (usedCoreNum > coreNum) ? coreNum : usedCoreNum;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // UB split: one float32 input tile plus one float32 output tile, budgeted
    // with double buffering to remain safe for both single- and double-buffer
    // kernel implementations.
    const int64_t elementCapacity = FloorDiv(static_cast<int64_t>(ubSize), TYPE_SIZE);
    int64_t ubFactor = FloorAlign(FloorDiv(elementCapacity, BUFFER_TENSOR_COUNT), ubBlockSize);
    if (ubFactor < MIN_UB_FACTOR) {
        ubFactor = MIN_UB_FACTOR;
    }
    tiling->ubFactor = ubFactor;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSpacing([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SpacingCompileInfo {};

IMPL_OP_OPTILING(Spacing)
    .Tiling(SpacingTilingFunc)
    .TilingParse<SpacingCompileInfo>(TilingParseForSpacing);

} // namespace optiling