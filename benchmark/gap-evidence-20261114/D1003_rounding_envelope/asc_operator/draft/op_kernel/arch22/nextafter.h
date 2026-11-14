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
 * \file nextafter.h
 * \brief Nextafter 算子 kernel 类定义（arch22 架构）
 *
 * 语义：
 *   y = nextafter(x, y)
 *   - 按 IEEE-754 float32 的相邻可表示值移动
 *   - x == y 时返回 y（保留 y 的符号，包括有符号零）
 *   - x 为 ±0 时返回带 y 符号的最小非规格化数
 *   - NaN 输入返回 NaN，不静默回退 CPU
 *   - ±Inf、非规格化数、有符号零均按 NumPy oracle 处理
 *
 * 当前候选仅支持 float32、同 shape、C 连续、无广播的输入组合。
 */

#ifndef NEXTAFTER_ARCH22_KERNEL_H
#define NEXTAFTER_ARCH22_KERNEL_H

#include <cstdint>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "nextafter_tiling_data.h"

namespace NsNextafter {

using namespace AscendC;

union FloatBitsUnion {
    float f;
    uint32_t u;
};

__aicore__ inline uint32_t FloatToBits(float value)
{
    FloatBitsUnion tmp;
    tmp.f = value;
    return tmp.u;
}

__aicore__ inline float BitsToFloat(uint32_t value)
{
    FloatBitsUnion tmp;
    tmp.u = value;
    return tmp.f;
}

__aicore__ inline bool IsNanBits(uint32_t bits)
{
    return ((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0u);
}

__aicore__ inline bool SameFloatBits(uint32_t xBits, uint32_t yBits)
{
    if (xBits == yBits) {
        return true;
    }
    // +0.0 与 -0.0 在 NumPy/C nextafter 中视为相等。
    return ((xBits & 0x7FFFFFFFu) == 0u) && ((yBits & 0x7FFFFFFFu) == 0u);
}

__aicore__ inline uint32_t FloatOrderKey(uint32_t bits)
{
    // 将 IEEE-754 float32 位模式映射为保持数值顺序的无符号整数键。
    return (bits & 0x80000000u) ? (~bits) : (bits | 0x80000000u);
}

__aicore__ inline float NextafterScalar(float x, float y)
{
    uint32_t xBits = FloatToBits(x);
    uint32_t yBits = FloatToBits(y);

    if (IsNanBits(xBits)) {
        return x;
    }
    if (IsNanBits(yBits)) {
        return y;
    }

    if (SameFloatBits(xBits, yBits)) {
        return y;
    }

    // x 为 ±0：返回带 y 符号的最小非规格化数。
    if ((xBits & 0x7FFFFFFFu) == 0u) {
        return BitsToFloat((yBits & 0x80000000u) | 0x00000001u);
    }

    // ±Inf 需要显式处理，避免对 Inf 位模式直接加减产生 NaN。
    if (xBits == 0x7F800000u) {
        return BitsToFloat(0x7F7FFFFFu);
    }
    if (xBits == 0xFF800000u) {
        return BitsToFloat(0xFF7FFFFFu);
    }

    bool towardPositive = FloatOrderKey(yBits) > FloatOrderKey(xBits);
    bool xNegative = (xBits & 0x80000000u) != 0u;

    uint32_t resultBits;
    if ((!xNegative && towardPositive) || (xNegative && !towardPositive)) {
        resultBits = xBits + 1u;
    } else {
        resultBits = xBits - 1u;
    }
    return BitsToFloat(resultBits);
}

template <typename T, int BUFFER_MODE = 0>
class Nextafter {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Nextafter() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const NextafterTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inputQueueX;
    TQue<QuePosition::VECIN, 2> inputQueueY;
    TQue<QuePosition::VECOUT, 2> outputQueueZ;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Nextafter<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                       const NextafterTilingData* tilingData)
{
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t offset = tilingData->blockFactor * blockIdx;
    int64_t remainderLength = tilingData->totalNum - offset;
    if (remainderLength < 0) {
        remainderLength = 0;
    }
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
    inputGMY.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + offset, blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Nextafter<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.template AllocTensor<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = static_cast<T>(0);

    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, padParams);
    AscendC::DataCopyPad(yLocal, inputGMY[progress * ubLength_], copyParams, padParams);

    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Nextafter<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.template DeQue<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    AscendC::DataCopyPad(outputGMZ[progress * ubLength_], zLocal, copyParams);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Nextafter<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.template DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.template AllocTensor<T>();

    uint32_t count = static_cast<uint32_t>(currentNum);
    for (uint32_t i = 0u; i < count; ++i) {
        float xValue = static_cast<float>(xLocal.GetValue(i));
        float yValue = static_cast<float>(yLocal.GetValue(i));
        float result = NextafterScalar(xValue, yValue);
        zLocal.SetValue(i, static_cast<T>(result));
    }

    outputQueueZ.template EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Nextafter<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsNextafter

#endif // NEXTAFTER_ARCH22_KERNEL_H