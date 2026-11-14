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
 * \file diff_tiling.cpp
 * \brief Diff operator tiling implementation for Ascend910B (arch22).
 *
 * This candidate is intentionally limited to the verified draft scope:
 *   - float32 input and output
 *   - static shape dimensions
 *   - ND contiguous tensors
 *   - NumPy diff signature with a single positional input and an optional axis keyword
 */

#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/diff_tiling_data.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1U;
constexpr int64_t TYPE_SIZE = static_cast<int64_t>(sizeof(float));
// Two input streams plus one output stream, with double buffering.
constexpr int64_t UB_TENSOR_COUNT = 6;
constexpr int64_t MIN_UB_FACTOR = 1;
constexpr int64_t FALLBACK_UB_BLOCK_SIZE = 32;

static bool SafeMul(int64_t a, int64_t b, int64_t* out)
{
    if (out == nullptr) {
        return false;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a < 0 || b < 0) {
        return false;
    }
    if (a > std::numeric_limits<int64_t>::max() / b) {
        return false;
    }
    *out = a * b;
    return true;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, ubSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, coreNum);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum <= 0, OP_LOGE(context, "Diff: coreNum is invalid: %ld", *coreNum), return ge::GRAPH_FAILED);

    *ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0U, OP_LOGE(context, "Diff: ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAxis(gert::TilingContext* context, int64_t rank, int64_t* axis)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    int64_t axisValue = -1;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(0);
        if (axisPtr != nullptr) {
            axisValue = *axisPtr;
        }
    }

    if (axisValue < 0) {
        axisValue += rank;
    }

    OP_CHECK_IF(
        axisValue < 0 || axisValue >= rank,
        OP_LOGE(context, "Diff: axis %ld is out of range for rank %ld", axisValue, rank),
        return ge::GRAPH_FAILED);

    *axis = axisValue;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t* totalNum, int64_t* outerNum, int64_t* innerNum, int64_t* axisLen,
    int64_t* outAxisLen, ge::DataType* dataType)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, totalNum);
    OP_CHECK_NULL_WITH_CONTEXT(context, outerNum);
    OP_CHECK_NULL_WITH_CONTEXT(context, innerNum);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisLen);
    OP_CHECK_NULL_WITH_CONTEXT(context, outAxisLen);
    OP_CHECK_NULL_WITH_CONTEXT(context, dataType);

    const gert::Shape* inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();

    const int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());
    OP_CHECK_IF(rank <= 0, OP_LOGE(context, "Diff: input rank must be at least 1, got %ld", rank), return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < rank; ++i) {
        const int64_t dim = inputShape.GetDim(static_cast<size_t>(i));
        OP_CHECK_IF(
            dim < 0,
            OP_LOGE(context, "Diff: dynamic input dim[%ld] is not supported: %ld", i, dim),
            return ge::GRAPH_FAILED);
    }

    int64_t axis = -1;
    OP_CHECK_IF(
        GetAxis(context, rank, &axis) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Diff: failed to resolve axis"),
        return ge::GRAPH_FAILED);

    const int64_t inputAxisLen = inputShape.GetDim(static_cast<size_t>(axis));
    const int64_t outputAxisLen = (inputAxisLen > 0) ? (inputAxisLen - 1) : 0;

    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) {
        int64_t nextOuter = 0;
        OP_CHECK_IF(
            !SafeMul(outer, inputShape.GetDim(static_cast<size_t>(i)), &nextOuter),
            OP_LOGE(context, "Diff: overflow while computing outer dimension"),
            return ge::GRAPH_FAILED);
        outer = nextOuter;
    }

    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) {
        int64_t nextInner = 0;
        OP_CHECK_IF(
            !SafeMul(inner, inputShape.GetDim(static_cast<size_t>(i)), &nextInner),
            OP_LOGE(context, "Diff: overflow while computing inner dimension"),
            return ge::GRAPH_FAILED);
        inner = nextInner;
    }

    int64_t outputNum = 0;
    int64_t tmpNum = 0;
    OP_CHECK_IF(
        !SafeMul(outer, outputAxisLen, &tmpNum),
        OP_LOGE(context, "Diff: overflow while computing output element count"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !SafeMul(tmpNum, inner, &outputNum),
        OP_LOGE(context, "Diff: overflow while computing output element count"),
        return ge::GRAPH_FAILED);

    const gert::Shape* outputShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShapePtr);
    const gert::Shape& outputShape = outputShapePtr->GetStorageShape();

    OP_CHECK_IF(
        outputShape.GetDimNum() != static_cast<size_t>(rank),
        OP_LOGE(
            context, "Diff: output rank mismatch, expected %ld, got %zu", rank,
            outputShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < rank; ++i) {
        const int64_t expectedDim = (i == axis) ? outputAxisLen : inputShape.GetDim(static_cast<size_t>(i));
        const int64_t actualDim = outputShape.GetDim(static_cast<size_t>(i));
        OP_CHECK_IF(
            actualDim != expectedDim,
            OP_LOGE(
                context, "Diff: output shape mismatch at dim %ld, expected %ld, got %ld", i, expectedDim,
                actualDim),
            return ge::GRAPH_FAILED);
    }

    const gert::CompileTimeTensorDesc* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        *dataType != ge::DT_FLOAT,
        OP_LOGE(context, "Diff: unsupported input dtype %d, only float32 is supported", static_cast<int32_t>(*dataType)),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(
        outputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(
            context, "Diff: unsupported output dtype %d, only float32 is supported",
            static_cast<int32_t>(outputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    *totalNum = outputNum;
    *outerNum = outer;
    *innerNum = inner;
    *axisLen = inputAxisLen;
    *outAxisLen = outputAxisLen;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DiffTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Diff: GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    int64_t outerNum = 0;
    int64_t innerNum = 0;
    int64_t axisLen = 0;
    int64_t outAxisLen = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalNum, &outerNum, &innerNum, &axisLen, &outAxisLen, &dataType) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Diff: GetShapeAttrsInfo failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Diff: GetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);

    DiffTilingData* tiling = context->GetTilingData<DiffTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(DiffTilingData), 0, sizeof(DiffTilingData)) != EOK,
        OP_LOGE(context, "Diff: failed to clear tiling data"),
        return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->outerNum = outerNum;
    tiling->innerNum = innerNum;
    tiling->axisLen = axisLen;
    tiling->outAxisLen = outAxisLen;

    if (totalNum == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(0);
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    if (ubBlockSize <= 0) {
        ubBlockSize = FALLBACK_UB_BLOCK_SIZE;
    }

    // Split output elements evenly across AIV cores. blockFactor is the
    // per-core output element count, rounded up to the UB block granularity.
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), ubBlockSize);
    if (blockFactor <= 0) {
        blockFactor = 1;
    }

    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    // Each output element reads two input elements and writes one output
    // element. Reserve six float buffers to remain safe with double buffering.
    const int64_t ubSizeElements = static_cast<int64_t>(ubSize) / TYPE_SIZE;
    int64_t ubFactor = FloorAlign(FloorDiv(ubSizeElements, UB_TENSOR_COUNT), ubBlockSize);
    if (ubFactor < MIN_UB_FACTOR) {
        ubFactor = MIN_UB_FACTOR;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForDiff([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct DiffCompileInfo {};

IMPL_OP_OPTILING(Diff).Tiling(DiffTilingFunc).TilingParse<DiffCompileInfo>(TilingParseForDiff);

} // namespace optiling