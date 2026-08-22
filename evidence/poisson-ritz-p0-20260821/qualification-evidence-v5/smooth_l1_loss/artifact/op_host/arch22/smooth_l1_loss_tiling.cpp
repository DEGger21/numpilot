/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_tiling.cpp
 * \brief SmoothL1Loss (beta=1, mean reduction) tiling implementation for ascend910b.
 *
 * Semantics (NumPy oracle):
 *   error = predict - label
 *   absolute = |error|
 *   bounded = min(absolute, 1.0)
 *   loss_elem = bounded * bounded / 2.0 + (absolute - bounded)
 *   output = mean(loss_elem)   // scalar float32
 *
 * Strategy: single AIV core. blockDim = 1, blockFactor = totalNum, the kernel
 * performs the full reduction on one core and writes the scalar directly
 * (no atomic add, no multicore reduction).
 *
 * Note: the internal registry OpType is NumPilotSmoothMeanLoss: a globally
 * unique, digit-free identifier. CANN 9.0's build-time and runtime
 * CamelCase-to-file-name conversion disagree around embedded digits (e.g. the
 * "L1" in NumPilotSmoothL1Loss), which caused the runtime to look for
 * num_pilot_smooth_l_1_loss.json / num_pilot_smooth_l1loss.json while the
 * build installed num_pilot_smooth_l1_loss.json. A digit-free OpType converts
 * consistently on both sides. The public API, file names, and ACLNN symbol
 * names remain smooth_l1_loss based.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/smooth_l1_loss_tiling_data.h"
#include "../../op_kernel/arch22/smooth_l1_loss_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4; // float32
constexpr size_t WORKSPACE_NUM = 1;
// Kernel UB tensors (single buffer): predict, label, absolute, bounded, reduceDst
// plus a 32-byte output-local buffer. Conservative count keeps every buffer
// ubFactor-sized (output-local is fixed 32B and fits in the margin).
constexpr int64_t SINGLE_BUF_TENSOR_COUNT = 5;
// Conservative fraction of UB actually handed to the tiling calculation.
constexpr int64_t UB_RESERVE_NUM = 2;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

// Get platform info: ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum) {
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Get shape/dtype info; validate rank>=1, float32, matching inputs, scalar output
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t* totalIdx, ge::DataType* dataType) {
    auto inputPredict = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputPredict);
    auto predictShape = EnsureNotScalar(inputPredict->GetStorageShape());
    auto inputLabel = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLabel);
    auto labelShape = EnsureNotScalar(inputLabel->GetStorageShape());
    auto outLoss = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outLoss);
    auto outShape = EnsureNotScalar(outLoss->GetStorageShape());

    OP_CHECK_IF(
        predictShape.GetShapeSize() != labelShape.GetShapeSize(),
        OP_LOGE(context, "NumPilotSmoothMeanLoss: predict and label shape size mismatch: predict=%ld, label=%ld",
            predictShape.GetShapeSize(), labelShape.GetShapeSize()),
        return ge::GRAPH_FAILED);
    // Output is a scalar (mean reduction)
    OP_CHECK_IF(
        outShape.GetShapeSize() != 1,
        OP_LOGE(context, "NumPilotSmoothMeanLoss: output must be scalar, got size=%ld", outShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    *totalIdx = predictShape.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(supportedDtype.count(*dataType) == 0, OP_LOGE(context, "NumPilotSmoothMeanLoss: invalid dtype, only float32 supported"), return ge::GRAPH_FAILED);
    auto labelDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelDesc);
    OP_CHECK_IF(
        labelDesc->GetDataType() != *dataType,
        OP_LOGE(context, "NumPilotSmoothMeanLoss: predict and label dtype mismatch"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Workspace must be declared even when size is 0
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context) {
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// Tiling entry
static ge::graphStatus SmoothL1LossTilingFunc(gert::TilingContext* context) {
    // 1. Platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "NumPilotSmoothMeanLoss: GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 2. Shape/dtype info
    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalIdx, &dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "NumPilotSmoothMeanLoss: GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // 3. Workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "NumPilotSmoothMeanLoss: GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4. Tiling data
    SmoothL1LossTilingData* tiling = context->GetTilingData<SmoothL1LossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SmoothL1LossTilingData), 0, sizeof(SmoothL1LossTilingData)) != EOK,
        OP_LOGE(context, "NumPilotSmoothMeanLoss: set tiling data error"), return ge::GRAPH_FAILED);

    // Empty tensor guard: kernel writes mean of zero elements as 0.0 on core 0
    if (totalIdx == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 1;
        tiling->coreNum = 1;
        tiling->finalReduce = 0;
        context->SetBlockDim(1);
        // float32 + single buffer mode
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT), 0U);
        return ge::GRAPH_SUCCESS;
    }

    // Single-core strategy: one AIV core processes the whole tensor and
    // performs the complete reduction; the kernel writes the final scalar
    // directly, so no atomic add or cross-core reduction is needed.
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = totalIdx;
    tiling->coreNum = 1;
    tiling->finalReduce = 1;

    // Conservative ubFactor: reserve headroom (UB_RESERVE_NUM) so every kernel
    // UB buffer (predict, label, absolute, bounded, reduceDst) can be
    // independently ubFactor-sized, plus the 32-byte output-local buffer.
    int64_t usableUbElems = FloorDiv(static_cast<int64_t>(ubSize), TYPE_SIZE * UB_RESERVE_NUM);
    tiling->ubFactor = FloorAlign(FloorDiv(usableUbElems, SINGLE_BUF_TENSOR_COUNT), ubBlockSize);
    OP_CHECK_IF(
        tiling->ubFactor <= 0,
        OP_LOGE(context, "NumPilotSmoothMeanLoss: ubFactor is invalid"), return ge::GRAPH_FAILED);

    context->SetBlockDim(1);

    // 5. TilingKey: single admitted dtype (float32), single buffer mode only
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT), 0U);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSmoothL1Loss([[maybe_unused]] gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

struct NumPilotSmoothMeanLossCompileInfo {}; // required for graph-mode integration

IMPL_OP_OPTILING(NumPilotSmoothMeanLoss)
    .Tiling(SmoothL1LossTilingFunc)
    .TilingParse<NumPilotSmoothMeanLossCompileInfo>(TilingParseForSmoothL1Loss);

} // namespace optiling