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
 * \file spacing.h
 * \brief Spacing operator kernel class definition for arch22 (Ascend910B).
 *
 * Semantic contract for this bounded candidate:
 *   - input dtype: float32 only
 *   - output dtype: float32
 *   - NumPy oracle: np.spacing(x) = np.nextafter(x, +inf) - x for finite non-zero x,
 *     smallest positive subnormal for zero/subnormal x, and NaN for infinities/NaN.
 */

#ifndef SPACING_H
#define SPACING_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "spacing_tiling_data.h"
#include "spacing_tiling_key.h"

namespace NsSpacing {

using namespace AscendC;

template <typename T = float, int BUFFER_MODE = 1>
class Spacing {
    static_assert(sizeof(T) == sizeof(float), "Spacing only supports float32");
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Spacing() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SpacingTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

    __aicore__ inline float SpacingValue(float x);
    __aicore__ inline uint32_t FloatToBits(float v);
    __aicore__ inline float BitsToFloat(uint32_t u);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueueX;
    TQue<QuePosition::VECOUT, 1> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Spacing<T, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, const SpacingTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    if (remainderLength < 0) {
        remainderLength = 0;
    }
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Spacing<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(
        xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, static_cast<T>(0)});
    inputQueueX.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Spacing<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Spacing<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    for (int64_t i = 0; i < currentNum; ++i) {
        float v = static_cast<float>(xLocal.GetValue(static_cast<uint32_t>(i)));
        float s = SpacingValue(v);
        yLocal.SetValue(static_cast<uint32_t>(i), static_cast<T>(s));
    }

    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Spacing<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline uint32_t Spacing<T, BUFFER_MODE>::FloatToBits(float v)
{
    union {
        float f;
        uint32_t u;
    } c;
    c.f = v;
    return c.u;
}

template <typename T, int BUFFER_MODE>
__aicore__ inline float Spacing<T, BUFFER_MODE>::BitsToFloat(uint32_t u)
{
    union {
        float f;
        uint32_t u;
    } c;
    c.u = u;
    return c.f;
}

template <typename T, int BUFFER_MODE>
__aicore__ inline float Spacing<T, BUFFER_MODE>::SpacingValue(float x)
{
    uint32_t u = FloatToBits(x);
    uint32_t exp = (u >> 23) & 0xFFu;
    uint32_t mant = u & 0x7FFFFFu;

    // Inf and NaN both map to NaN, matching NumPy's npy_spacing.
    if (exp == 0xFFu) {
        return BitsToFloat(0x7FC00000u);
    }

    // Zero and subnormal values: nextafter(x, +inf) - x is the smallest
    // subnormal.  The sign is preserved for negative subnormals; signed zero
    // still returns the positive smallest subnormal, matching np.spacing.
    if (exp == 0u) {
        if ((u & 0x7FFFFFFFu) == 0u) {
            return BitsToFloat(0x00000001u);
        }
        return BitsToFloat(0x00000001u | (u & 0x80000000u));
    }

    int32_t e = static_cast<int32_t>(exp);
    int32_t k;
    if ((u & 0x80000000u) != 0u) {
        if (mant == 0u) {
            // Exact negative power of two: nextafter(x, +inf) moves into the
            // next lower magnitude binade, so the spacing is half the normal ULP
            // except at the smallest normal binade, where subnormal steps apply.
            if (e == 1) {
                k = -149;
            } else {
                k = e - 151;
            }
        } else {
            k = e - 150;
        }
    } else {
        k = e - 150;
    }

    uint32_t resultBits;
    if (k >= -126) {
        resultBits = static_cast<uint32_t>(k + 127) << 23;
    } else {
        resultBits = 1u << (k + 149);
    }
    // np.spacing is nextafter(x, +inf) - x, so finite negative inputs produce
    // a negative spacing.  The magnitude above is unsigned.
    if ((u & 0x80000000u) != 0u) {
        resultBits |= 0x80000000u;
    }
    return BitsToFloat(resultBits);
}

} // namespace NsSpacing

#endif // SPACING_H