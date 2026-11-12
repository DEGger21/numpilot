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
 * \file tril_tiling.cpp
 * \brief Host tiling for Tril (lower-triangular part of a 2-D float32 tensor).
 *
 * NumPy semantics implemented by this draft:
 *     numpy.tril(a)  ->  asnumpy.tril(a)
 * with the default diagonal k = 0.  Only the measured float32, rank-2,
 * C-contiguous profile is admitted.  Every other dtype/rank is rejected
 * explicitly; no implicit cast and no host fallback is performed.
 */

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/tril_tiling_data.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

namespace {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;    // float32
constexpr size_t WORKSPACE_NUM = 1U;
constexpr int64_t BUFFER_COUNT = 8; // conservative UB tensor/buffer budget

} // namespace

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(*coreNum <= 0, OP_LOGE(context, "Tril: coreNum is %ld", *coreNum), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0U, OP_LOGE(context, "Tril: ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t* totalIdx, int64_t* rows,
                                         int64_t* cols, ge::DataType* dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    const gert::Shape& inputShape = inputX->GetStorageShape();
    OP_CHECK_IF(inputShape.GetDimNum() != 2U,
                OP_LOGE(context, "Tril: input must be rank-2, got rank %d", static_cast<int>(inputShape.GetDimNum())),
                return ge::GRAPH_FAILED);

    *rows = inputShape.GetDim(0);
    *cols = inputShape.GetDim(1);
    OP_CHECK_IF(*rows < 0 || *cols < 0, OP_LOGE(context, "Tril: negative dimension rows=%ld cols=%ld", *rows, *cols),
                return ge::GRAPH_FAILED);

    auto outputY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputY);
    const gert::Shape& outputShape = outputY->GetStorageShape();
    OP_CHECK_IF(outputShape.GetDimNum() != 2U,
                OP_LOGE(context, "Tril: output must be rank-2, got rank %d", static_cast<int>(outputShape.GetDimNum())),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputShape.GetDim(0) != *rows || outputShape.GetDim(1) != *cols,
                OP_LOGE(context, "Tril: output shape [%ld,%ld] does not match input shape [%ld,%ld]",
                        outputShape.GetDim(0), outputShape.GetDim(1), *rows, *cols),
                return ge::GRAPH_FAILED);

    const int64_t kInt64Max = std::numeric_limits<int64_t>::max();
    OP_CHECK_IF(*rows != 0 && *cols > kInt64Max / *rows,
                OP_LOGE(context, "Tril: element count overflows int64 rows=%ld cols=%ld", *rows, *cols),
                return ge::GRAPH_FAILED);
    *totalIdx = *rows * *cols;

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(*dataType != ge::DT_FLOAT,
                OP_LOGE(context, "Tril: only float32 input is supported, got dtype %d", static_cast<int>(*dataType)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TrilTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Tril: GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    int64_t totalIdx = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &totalIdx, &rows, &cols, &dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Tril: GetShapeAttrsInfo failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Tril: GetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    TrilTilingData* tiling = context->GetTilingData<TrilTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    std::memset(static_cast<void*>(tiling), 0, sizeof(TrilTilingData));

    tiling->totalNum = totalIdx;
    tiling->rows = rows;
    tiling->cols = cols;

    if (totalIdx == 0) {
        context->SetBlockDim(1U);
        context->SetTilingKey(0U);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context, "Tril: invalid ubBlockSize %ld", ubBlockSize),
                return ge::GRAPH_FAILED);

    const int64_t blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    const int64_t usedCoreNum = CeilDiv(totalIdx, blockFactor);
    OP_CHECK_IF(usedCoreNum <= 0 || usedCoreNum > coreNum,
                OP_LOGE(context, "Tril: invalid usedCoreNum %ld (coreNum=%ld)", usedCoreNum, coreNum),
                return ge::GRAPH_FAILED);

    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / TYPE_SIZE, BUFFER_COUNT), ubBlockSize);
    if (ubFactor <= 0) {
        ubFactor = ubBlockSize;
    }
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "Tril: invalid ubFactor"), return ge::GRAPH_FAILED);

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->usedCoreNum = usedCoreNum;
    tiling->alignCols = CeilAlign(cols, ubBlockSize);

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0U);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTril([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TrilCompileInfo {};

IMPL_OP_OPTILING(Tril).Tiling(TrilTilingFunc).TilingParse<TrilCompileInfo>(TilingParseForTril);

} // namespace optiling