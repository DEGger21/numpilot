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
 * \file ediff1d_tiling.cpp
 * \brief Ediff1d host tiling for Ascend910B3 (arch22).
 *
 * The supported kernel computes the flattened first difference:
 *   y[i] = x[i + 1] - x[i], for i in [0, input_size - 2].
 *
 * Only float32 input/output is admitted by this bounded candidate. Other
 * dtypes are rejected rather than silently cast. Shapes are validated against
 * the shape inferred by the host infershape stage, and the output length must
 * be max(input_size - 1, 0).
 */

#include <cstdint>
#include <cstring>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

#if __has_include("ediff1d_tiling_data.h")
#include "ediff1d_tiling_data.h"
#elif __has_include("../../op_kernel/arch22/ediff1d_tiling_data.h")
#include "../../op_kernel/arch22/ediff1d_tiling_data.h"
#else
#error "ediff1d_tiling_data.h must be provided by the kernel/arch22 stage"
#endif

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1U;
constexpr int64_t FLOAT32_SIZE = 4;
constexpr int64_t EDIFF1D_MAX_UB_FACTOR = 4096;
constexpr int64_t BUFFER_NUM = 2;  // one input tile and one output tile
constexpr uint32_t SINGLE_TILING_KEY = 0U;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(*coreNum <= 0, OP_LOGE(context, "Ediff1d: coreNum is invalid"), return ge::GRAPH_FAILED);

    *ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0U, OP_LOGE(context, "Ediff1d: ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDesc(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_IF(
        inputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(context, "Ediff1d: only float32 input is supported, got dtype %d",
                static_cast<int32_t>(inputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(
        outputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(context, "Ediff1d: only float32 output is supported, got dtype %d",
                static_cast<int32_t>(outputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, int64_t* inputNum, int64_t* outputNum)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    const gert::Shape inputStorageShape = EnsureNotScalar(inputShape->GetStorageShape());
    const gert::Shape outputStorageShape = EnsureNotScalar(outputShape->GetStorageShape());

    *inputNum = inputStorageShape.GetShapeSize();
    OP_CHECK_IF(*inputNum < 0, OP_LOGE(context, "Ediff1d: invalid negative input size"), return ge::GRAPH_FAILED);

    *outputNum = (*inputNum > 0) ? (*inputNum - 1) : 0;

    const int64_t declaredOutputNum = outputStorageShape.GetShapeSize();
    OP_CHECK_IF(
        declaredOutputNum != *outputNum,
        OP_LOGE(context, "Ediff1d: output shape size mismatch, expected %ld, got %ld",
                *outputNum, declaredOutputNum),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Ediff1dTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Ediff1d: GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    int64_t inputNum = 0;
    int64_t outputNum = 0;
    OP_CHECK_IF(
        GetShapeInfo(context, &inputNum, &outputNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Ediff1d: GetShapeInfo failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckTensorDesc(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Ediff1d: CheckTensorDesc failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Ediff1d: SetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);

    Ediff1dTilingData* tiling = context->GetTilingData<Ediff1dTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        std::memset(tiling, 0, sizeof(Ediff1dTilingData)) == nullptr,
        OP_LOGE(context, "Ediff1d: failed to clear tiling data"),
        return ge::GRAPH_FAILED);

    if (outputNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1U);
        context->SetTilingKey(SINGLE_TILING_KEY);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t ubBlockSize = GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context, "Ediff1d: invalid UB block size"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ubSize < static_cast<uint64_t>(BUFFER_NUM * ubBlockSize * FLOAT32_SIZE),
        OP_LOGE(context, "Ediff1d: UB size is too small for the minimum tile"),
        return ge::GRAPH_FAILED);

    const int64_t blockFactor = CeilAlign(CeilDiv(outputNum, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(outputNum, blockFactor);
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    // Three queues are live at once in the kernel (two shifted input tiles
    // plus the output tile) and each holds BUFFER_NUM buffers, so the UB
    // budget must account for all of them.
    constexpr int64_t QUEUE_COUNT = 3;
    int64_t ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize) / FLOAT32_SIZE, BUFFER_NUM * QUEUE_COUNT),
        ubBlockSize);
    if (ubFactor < ubBlockSize) {
        ubFactor = ubBlockSize;
    }
    ubFactor = std::min(ubFactor, EDIFF1D_MAX_UB_FACTOR);

    tiling->totalNum = outputNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(SINGLE_TILING_KEY);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForEdiff1d([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct Ediff1dCompileInfo {};

IMPL_OP_OPTILING(Ediff1d)
    .Tiling(Ediff1dTilingFunc)
    .TilingParse<Ediff1dCompileInfo>(TilingParseForEdiff1d);

}  // namespace optiling