/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file apply_gradient_descent_tiling.cpp
 * \brief ApplyGradientDescent 算子 Ascend910B (arch22) Tiling 实现
 *        语义：out = var - alpha * delta (非原地, float32)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/apply_gradient_descent_tiling_data.h"
#include "../../op_kernel/arch22/apply_gradient_descent_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4; // float32
constexpr size_t WORKSPACE_NUM = 1;
// 单缓冲: 3输入(var, alpha, delta) x1 + 1输出 x1 = 4 个 UB tensor
constexpr int64_t SINGLE_BUF_TENSOR_COUNT = 4;
// 双缓冲: 3输入x2 + 1输出x2 = 8 个 UB tensor
constexpr int64_t DOUBLE_BUF_TENSOR_COUNT = 8;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t* totalIdx,
                                         ge::DataType* dataType)
{
    // 输入顺序与算子定义一致: var(0), alpha(1), delta(2); 输出: out(0)
    auto varShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    auto deltaShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, deltaShape);
    auto outShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);

    auto varS = EnsureNotScalar(varShape->GetStorageShape());
    auto deltaS = EnsureNotScalar(deltaShape->GetStorageShape());
    auto outS = EnsureNotScalar(outShapePtr->GetStorageShape());

    OP_CHECK_IF(
        varS.GetShapeSize() != deltaS.GetShapeSize() || varS.GetShapeSize() != outS.GetShapeSize(),
        OP_LOGE(context,
                "ApplyGradientDescent: shape size mismatch: var=%ld, delta=%ld, out=%ld",
                varS.GetShapeSize(), deltaS.GetShapeSize(), outS.GetShapeSize()),
        return ge::GRAPH_FAILED);

    *totalIdx = varS.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(supportedDtype.count(*dataType) == 0, OP_LOGE(context, "invalid dtype, only float32 supported"),
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

static ge::graphStatus ApplyGradientDescentTilingFunc(gert::TilingContext* context)
{
    // 1. 平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. shape / dtype 信息
    int64_t totalIdx = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &totalIdx, &dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. workspace 声明
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    // 4. 设置 tiling data (仅 totalNum / blockFactor / ubFactor)
    ApplyGradientDescentTilingData* tiling = context->GetTilingData<ApplyGradientDescentTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ApplyGradientDescentTilingData), 0,
                         sizeof(ApplyGradientDescentTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    if (totalIdx == 0) {
        context->SetBlockDim(1);
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, 0);
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? DOUBLE_BUF_TENSOR_COUNT : SINGLE_BUF_TENSOR_COUNT;
    tiling->ubFactor = FloorAlign(FloorDiv((static_cast<int64_t>(ubSize) / TYPE_SIZE), bufferNum), ubBlockSize);
    OP_CHECK_IF(tiling->ubFactor == 0, OP_LOGE(context, "ubFactor is 0"), return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);

    // 5. 设置 TilingKey: 参数顺序与 apply_gradient_descent_tiling_key.h 中
    //    ASCENDC_TPL_ARGS_DECL 定义一致，映射到 kernel 入口模板参数
    //    apply_gradient_descent<D_T_X, BUFFER_MODE>
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApplyGradientDescent([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyGradientDescentCompileInfo {}; // 必须定义, 入图场景依赖

IMPL_OP_OPTILING(ApplyGradientDescent)
    .Tiling(ApplyGradientDescentTilingFunc)
    .TilingParse<ApplyGradientDescentCompileInfo>(TilingParseForApplyGradientDescent);

} // namespace optiling