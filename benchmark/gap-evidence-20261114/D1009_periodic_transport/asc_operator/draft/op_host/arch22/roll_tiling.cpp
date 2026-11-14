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
 * \file roll_tiling.cpp
 * \brief Host tiling for the Roll operator (numpy.roll) on arch22 / Ascend910B.
 *
 * Declared support scope of this draft (see the GapSpec that generated it):
 *   - input dtype  : float32 only; every other dtype is rejected explicitly,
 *                    no implicit cast and no silent CPU fallback;
 *   - input layout : C contiguous ND storage;
 *   - shift / axis : int64 attributes, attribute 0 = shift, attribute 1 = axis.
 *                    The op definition must register the attributes in exactly
 *                    this order because gert::RuntimeAttrs is index based;
 *   - axis         : absent attribute (or the documented "no axis" sentinel)
 *                    means numpy.roll over the flattened array; an integer
 *                    axis must be inside [-rank, rank - 1] and is rejected for
 *                    rank-0 inputs, matching numpy.roll.
 *
 * Semantics: for a C-contiguous input viewed as [outerDim, axisDim, innerDim]
 * (row major), numpy.roll(x, shift, axis) is exactly
 *     y[o, a, i] = x[o, (a - shift) mod axisDim, i]
 * which, in flat memory, is two contiguous segment copies for every outer block:
 *     y[base .. base + headLen)            <- x[base + tailLen .. base + blockLen)
 *     y[base + headLen .. base + blockLen) <- x[base .. base + tailLen)
 * with blockLen = axisDim * innerDim, headLen = shiftNorm * innerDim and
 * tailLen = blockLen - headLen. The kernel consumes this segment description, so
 * no per-element index arithmetic is required on device and signed zeros, NaN,
 * +Inf and -Inf are transported bit-exactly (a pure permutation performs no
 * floating point arithmetic and cannot change signbit).
 *
 * TilingData contract shared with op_kernel/arch22/roll_tiling_data.h:
 *   totalNum      : total number of output elements
 *   blockLen      : elements per outer block (axisDim * innerDim)
 *   headLen       : elements of an outer block that wrap to the front
 *   tailLen       : blockLen - headLen
 *   outerDim      : number of outer blocks
 *   blocksPerCore : outer blocks assigned to a single AI vector core
 *   ubLen         : maximum number of float32 elements staged through UB per copy
 *
 * Empty tensors are accepted: totalNum == 0 is propagated and the kernel exits
 * without touching global memory. This draft targets static shapes only; a
 * dynamic shape (any negative dimension) is rejected explicitly instead of
 * being guessed, because the GapSpec leaves shape support unresolved.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/roll_tiling_data.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

namespace {
constexpr size_t WORKSPACE_NUM = 1U;
constexpr uint64_t WS_SYS_SIZE = 0U;
constexpr int64_t BYTES_PER_FLOAT = 4;
// The staging buffer only needs one UB tensor (GM -> UB -> GM), but leaving half
// of UB untouched keeps room for the queue descriptors the kernel allocates.
constexpr int64_t UB_BUFFER_NUM = 2;
// Documented encoding of "axis=None" when the optional attribute is materialized
// by the ACLNN adapter. An absent attribute is handled as None as well.
constexpr int64_t NO_AXIS_SENTINEL = std::numeric_limits<int64_t>::min();
// This draft registers exactly one TilingKey: the float32 segment-copy kernel.
constexpr uint64_t ROLL_TILING_KEY = 0U;
} // namespace

// Overflow checked multiplication of two non-negative dimensions.
static inline bool SafeMul(int64_t lhs, int64_t rhs, int64_t& out)
{
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

// Query the platform: usable UB bytes and the number of AI vector cores.
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(
        *coreNum <= 0, OP_LOGE(context, "Roll tiling: invalid AIV core number %ld", *coreNum),
        return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0U, OP_LOGE(context, "Roll tiling: invalid UB size 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Resolve dtype, shape decomposition and attributes into the roll layout.
// Decomposes the input shape into [outerDim, axisDim, innerDim] and normalizes
// the shift into [0, axisDim). Rejects everything outside the declared scope.
static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, int64_t* totalNum, int64_t* outerDim, int64_t* axisDim, int64_t* innerDim,
    int64_t* shiftNorm)
{
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    const gert::Shape& inputShape = inputStorageShape->GetStorageShape();

    const gert::StorageShape* outputStorageShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputStorageShape);
    const gert::Shape& outputShape = outputStorageShape->GetStorageShape();

    const int64_t inputNum = inputShape.GetShapeSize();
    OP_CHECK_IF(
        inputNum < 0,
        OP_LOGE(context, "Roll tiling: dynamic shape is outside the declared support scope (shape size %ld)", inputNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outputShape.GetShapeSize() != inputNum,
        OP_LOGE(
            context, "Roll tiling: output element count %ld does not match input element count %ld",
            outputShape.GetShapeSize(), inputNum),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_CHECK_IF(
        inputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(
            context, "Roll tiling: only float32 input is supported, got ge::DataType %d",
            static_cast<int32_t>(inputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(
        outputDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(
            context, "Roll tiling: only float32 output is supported, got ge::DataType %d",
            static_cast<int32_t>(outputDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // Attribute 0 is numpy.roll's required positional `shift`.
    const int64_t* shiftPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shiftPtr);
    const int64_t shift = *shiftPtr;
    // Attribute 1 is numpy.roll's optional keyword `axis`.
    const int64_t* axisPtr = attrs->GetAttrNum() > 1U ? attrs->GetAttrPointer<int64_t>(1) : nullptr;

    const int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());
    int64_t outer = 1;
    int64_t rolledLen = inputNum;
    int64_t inner = 1;

    if (axisPtr != nullptr && *axisPtr != NO_AXIS_SENTINEL) {
        OP_CHECK_IF(
            rank == 0,
            OP_LOGE(context, "Roll tiling: integer axis %ld is invalid for a rank-0 input", *axisPtr),
            return ge::GRAPH_FAILED);
        int64_t axis = *axisPtr;
        if (axis < 0) {
            axis += rank;
        }
        OP_CHECK_IF(
            axis < 0 || axis >= rank,
            OP_LOGE(context, "Roll tiling: axis %ld is out of range for rank %ld", *axisPtr, rank),
            return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < axis; ++i) {
            OP_CHECK_IF(
                !SafeMul(outer, inputShape.GetDim(i), outer),
                OP_LOGE(context, "Roll tiling: element count overflow before axis %ld", axis),
                return ge::GRAPH_FAILED);
        }
        rolledLen = inputShape.GetDim(axis);
        OP_CHECK_IF(
            rolledLen < 0, OP_LOGE(context, "Roll tiling: rolled axis length %ld is invalid", rolledLen),
            return ge::GRAPH_FAILED);
        for (int64_t i = axis + 1; i < rank; ++i) {
            OP_CHECK_IF(
                !SafeMul(inner, inputShape.GetDim(i), inner),
                OP_LOGE(context, "Roll tiling: element count overflow after axis %ld", axis),
                return ge::GRAPH_FAILED);
        }
    }

    int64_t expectedNum = 0;
    OP_CHECK_IF(
        !SafeMul(outer, rolledLen, expectedNum) || !SafeMul(expectedNum, inner, expectedNum),
        OP_LOGE(context, "Roll tiling: element count overflow for shape of rank %ld", rank),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        expectedNum != inputNum,
        OP_LOGE(
            context, "Roll tiling: decomposed element count %ld does not match input element count %ld",
            expectedNum, inputNum),
        return ge::GRAPH_FAILED);

    int64_t normalizedShift = 0;
    if (rolledLen > 0) {
        normalizedShift = shift % rolledLen;
        if (normalizedShift < 0) {
            normalizedShift += rolledLen;
        }
    }

    *totalNum = inputNum;
    *outerDim = outer;
    *axisDim = rolledLen;
    *innerDim = inner;
    *shiftNorm = normalizedShift;
    return ge::GRAPH_SUCCESS;
}

// Even with a zero sized workspace the slot has to be declared, otherwise the
// runtime does not hand a workspace pointer to the kernel.
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(WS_SYS_SIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RollTilingFunc(gert::TilingContext* context)
{
    // 1. Platform information (UB size, vector core count).
    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Roll tiling: failed to query platform information"), return ge::GRAPH_FAILED);

    // 2. Dtype, shape and attribute validation plus layout decomposition.
    int64_t totalNum = 0;
    int64_t outerDim = 0;
    int64_t axisDim = 0;
    int64_t innerDim = 0;
    int64_t shiftNorm = 0;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, &totalNum, &outerDim, &axisDim, &innerDim, &shiftNorm) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Roll tiling: failed to resolve shape and attributes"), return ge::GRAPH_FAILED);

    // 3. Workspace declaration.
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Roll tiling: failed to declare workspace"), return ge::GRAPH_FAILED);

    // 4. TilingData. The buffer is created by the kernel side
    //    REGISTER_TILING_DEFAULT(RollTilingData); zero it before filling so no
    //    stale field can leak into the kernel.
    RollTilingData* tiling = context->GetTilingData<RollTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    *tiling = RollTilingData();

    int64_t blockLen = 0;
    int64_t headLen = 0;
    int64_t tailLen = 0;
    int64_t blocksPerCore = 0;
    int64_t usedCoreNum = 1;
    int64_t ubLen = 0;

    if (totalNum > 0) {
        // For a static non-empty tensor every factor is strictly positive.
        blockLen = axisDim * innerDim;
        headLen = shiftNorm * innerDim;
        tailLen = blockLen - headLen;

        // Outer blocks are the parallel unit: one outer block is one independent
        // [axisDim, innerDim] plane, therefore cores never write the same range.
        const int64_t wantedCores = outerDim < coreNum ? outerDim : coreNum;
        usedCoreNum = wantedCores > 0 ? wantedCores : 1;
        blocksPerCore = CeilDiv(outerDim, usedCoreNum);
        usedCoreNum = CeilDiv(outerDim, blocksPerCore);

        // Staging buffer: half of UB, expressed in float32 elements and aligned
        // to the UB block granularity used by the copy API.
        int64_t ubBlockElements = static_cast<int64_t>(GetUbBlockSize(context)) / BYTES_PER_FLOAT;
        if (ubBlockElements < 1) {
            ubBlockElements = 1;
        }
        ubLen = FloorAlign(
            FloorDiv(static_cast<int64_t>(ubSize) / BYTES_PER_FLOAT, UB_BUFFER_NUM), ubBlockElements);
        if (ubLen < ubBlockElements) {
            ubLen = ubBlockElements;
        }
    }

    tiling->totalNum = totalNum;
    tiling->rowLen = blockLen;
    tiling->shiftOffset = headLen;
    tiling->rowCount = outerDim;
    tiling->blockRows = blocksPerCore;
    tiling->ubFactor = ubLen;

    OP_CHECK_IF(
        usedCoreNum <= 0, OP_LOGE(context, "Roll tiling: invalid block dim %ld", usedCoreNum),
        return ge::GRAPH_FAILED);

    // 5. Launch configuration. The draft registers a single TilingKey (0), i.e.
    //    the single float32 segment-copy kernel entry.
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(ROLL_TILING_KEY);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRoll([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct RollCompileInfo {}; // Required: graph mode tiling depends on this type.

// Tiling registration entry point.
IMPL_OP_OPTILING(Roll).Tiling(RollTilingFunc).TilingParse<RollCompileInfo>(TilingParseForRoll);

} // namespace optiling