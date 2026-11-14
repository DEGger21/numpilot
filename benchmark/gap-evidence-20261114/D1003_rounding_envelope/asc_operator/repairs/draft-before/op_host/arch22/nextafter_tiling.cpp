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
 * \file nextafter_tiling.cpp
 * \brief Nextafter 算子 tiling 实现（arch22 / Ascend910B3）
 *
 * 语义：out = numpy.nextafter(x1, x2)，逐元素返回 x1 朝 x2 方向的下一个可表示 float32 值。
 *
 * 本候选的支持范围严格限定在已观测 profile 内：
 *   - x1、x2 均为 float32；
 *   - x1、x2、out 形状完全一致（不支持广播），C 连续；
 *   - 调用签名固定为两个位置参数（keyword 一律不接受，由 Python 适配层拒绝）。
 * 范围外的 dtype / 广播形状在本文件中显式报错返回 GRAPH_FAILED，不做隐式 cast，
 * 也不存在任何 CPU/NumPy 回退路径。
 */

#include <cstdint>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/nextafter_tiling_data.h"
#include "../../op_kernel/arch22/nextafter_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

namespace {
constexpr size_t WORKSPACE_NUM = 1U;
constexpr size_t WS_SYS_SIZE = 0U;
// float32 单元素字节数
constexpr int64_t TYPE_SIZE = 4;
// UB 内同时驻留的 float32 Tensor 数量估算（用于切分 ubFactor）：
//   2 个输入 + 1 个输出 + 3 个计算临时量 = 6（单缓冲），双缓冲乒乓翻倍 = 12。
// 该估算偏保守，只影响单次搬运粒度，不影响数值正确性。
constexpr int64_t SINGLE_BUF_TENSOR_COUNT = 6;
constexpr int64_t DOUBLE_BUF_TENSOR_COUNT = 12;
// 规模小于该值时使用单缓冲，避免小算子的无谓开销
constexpr int64_t MIN_DOUBLE_BUF_ELEM_NUM = 1024;
// tiling_key 中唯一的 float32 模式取值（与 ASCENDC_TPL_ARGS_DECL 的顺序一致）
constexpr uint32_t SCH_MODE_FLOAT32 = 0U;
} // namespace

// 判断两个 shape 是否逐维完全一致（逐元素算子不支持广播，必须完全一致）
static bool IsSameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0U; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

// 获取平台信息：AIV 核数与 UB 大小
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        *coreNum <= 0, OP_LOGE(context, "Nextafter: aiv core num must be positive, but got %ld", *coreNum),
        return ge::GRAPH_FAILED);
    *ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(
        *ubSize == 0UL, OP_LOGE(context, "Nextafter: ub size must be positive"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取并校验 shape / dtype 信息，输出元素总数
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t* totalNum)
{
    auto inputShape1 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape1);
    auto inputShape2 = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape2);
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    const gert::Shape& shapeX1 = inputShape1->GetStorageShape();
    const gert::Shape& shapeX2 = inputShape2->GetStorageShape();
    const gert::Shape& shapeOut = outputShape->GetStorageShape();

    // 逐元素语义要求三者形状完全一致；广播属于明确拒绝的范围
    OP_CHECK_IF(
        !IsSameShape(shapeX1, shapeX2) || !IsSameShape(shapeX1, shapeOut),
        OP_LOGE(
            context,
            "Nextafter: x1/x2/out shapes must be identical (broadcast is not supported), "
            "dimNum=%zu/%zu/%zu, size=%ld/%ld/%ld",
            shapeX1.GetDimNum(), shapeX2.GetDimNum(), shapeOut.GetDimNum(), shapeX1.GetShapeSize(),
            shapeX2.GetShapeSize(), shapeOut.GetShapeSize()),
        return ge::GRAPH_FAILED);

    auto inputDesc1 = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc1);
    auto inputDesc2 = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc2);
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);

    // 仅支持 float32；其他 dtype 显式拒绝，绝不静默 cast
    const ge::DataType dtypeX1 = inputDesc1->GetDataType();
    const ge::DataType dtypeX2 = inputDesc2->GetDataType();
    const ge::DataType dtypeOut = outputDesc->GetDataType();
    OP_CHECK_IF(
        dtypeX1 != ge::DT_FLOAT,
        OP_LOGE(
            context, "Nextafter: only float32 input is supported, but input[0] dtype is %d",
            static_cast<int32_t>(dtypeX1)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dtypeX2 != ge::DT_FLOAT,
        OP_LOGE(
            context, "Nextafter: only float32 input is supported, but input[1] dtype is %d",
            static_cast<int32_t>(dtypeX2)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dtypeOut != ge::DT_FLOAT,
        OP_LOGE(
            context, "Nextafter: only float32 output is supported, but output[0] dtype is %d",
            static_cast<int32_t>(dtypeOut)),
        return ge::GRAPH_FAILED);

    *totalNum = shapeX1.GetShapeSize();
    OP_CHECK_IF(
        *totalNum < 0, OP_LOGE(context, "Nextafter: invalid shape size %ld", *totalNum), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 即使 workspace 大小为 0，也必须调用 GetWorkspaceSizes 声明 workspace slot，
// 否则运行时不会为 kernel 分配 workspace 指针。
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus NextafterTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息（核数、UB 大小）
    uint64_t ubSize = 0UL;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Nextafter: GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    // 2、获取并校验 shape / dtype 信息
    int64_t totalNum = 0;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Nextafter: GetShapeAttrsInfo failed"), return ge::GRAPH_FAILED);

    // 3、声明 workspace（本算子不需要额外 workspace，但仍需声明 slot）
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Nextafter: GetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    // 4、清零 TilingData（结构体定义由 kernel 侧 nextafter_tiling_data.h 唯一持有）
    NextafterTilingData* tiling = context->GetTilingData<NextafterTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(NextafterTilingData), 0, sizeof(NextafterTilingData)) != EOK,
        OP_LOGE(context, "Nextafter: failed to clear tiling data"), return ge::GRAPH_FAILED);

    // 5、空 Tensor 直接返回，避免后续除零
    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1U);
        ASCENDC_TPL_SEL_PARAM(context, SCH_MODE_FLOAT32);
        return ge::GRAPH_SUCCESS;
    }

    // 6、DMA 最小粒度：UB block 字节数换算为 float32 元素个数，至少为 1
    int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    OP_CHECK_IF(
        ubBlockSize <= 0, OP_LOGE(context, "Nextafter: ub block size must be positive, but got %ld", ubBlockSize),
        return ge::GRAPH_FAILED);
    int64_t blockAlign = ubBlockSize / TYPE_SIZE;
    if (blockAlign < 1) {
        blockAlign = 1;
    }

    // 7、多核切分：blockFactor 按 DMA 粒度向上对齐，保证相邻核写 GM 时不会相互越界覆盖
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), blockAlign);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // 8、UB 切分：按可用 UB 大小与驻留 Tensor 数量均分，再按 DMA 粒度向下对齐
    const int64_t useDoubleBuffer = (totalNum > MIN_DOUBLE_BUF_ELEM_NUM) ? 1 : 0;
    const int64_t tensorCount = (useDoubleBuffer == 1) ? DOUBLE_BUF_TENSOR_COUNT : SINGLE_BUF_TENSOR_COUNT;
    const int64_t ubElementNum = static_cast<int64_t>(ubSize) / TYPE_SIZE;
    const int64_t ubFactor = FloorAlign(FloorDiv(ubElementNum, tensorCount), blockAlign);
    OP_CHECK_IF(
        ubFactor <= 0,
        OP_LOGE(
            context, "Nextafter: invalid ubFactor %ld (ubSize=%lu, tensorCount=%ld, blockAlign=%ld)", ubFactor,
            static_cast<unsigned long>(ubSize), tensorCount, blockAlign),
        return ge::GRAPH_FAILED);
    tiling->ubFactor = ubFactor;

    // 9、设置启动核数
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 10、设置 TilingKey：参数个数与顺序必须与 ASCENDC_TPL_ARGS_DECL 完全一致（单参数 fp32 模式）
    ASCENDC_TPL_SEL_PARAM(context, SCH_MODE_FLOAT32);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForNextafter([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct NextafterCompileInfo {}; // 必须定义，入图场景依赖

// tiling 注册入口
IMPL_OP_OPTILING(Nextafter).Tiling(NextafterTilingFunc).TilingParse<NextafterCompileInfo>(TilingParseForNextafter);

} // namespace optiling