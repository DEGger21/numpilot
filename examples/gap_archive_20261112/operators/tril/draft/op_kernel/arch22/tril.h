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
 * \file tril.h
 * \brief Tril operator kernel class definition for arch22 (Ascend910B3).
 *
 * Semantics for this draft: out[i, j] = x[i, j] when j <= i, otherwise 0.0f.
 * This is numpy.tril(x) with the default diagonal k = 0 and one positional
 * tensor argument. The host-side validation admits float32 rank-2
 * C-contiguous tensors and rejects other dtypes/shapes without casting.
 */

#ifndef TRIL_ARCH22_H
#define TRIL_ARCH22_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tril_tiling_data.h"
#include "tril_tiling_key.h"

namespace NsTril {

using namespace AscendC;

template <typename T, int BUFFER_MODE = 1>
class Tril {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Tril() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TrilTilingData* tilingData)
    {
        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t blockFactor =
            (tilingData->blockFactor > 0) ? tilingData->blockFactor : tilingData->totalNum;

        const int64_t remainderLength = tilingData->totalNum - blockFactor * blockIdx;
        blockLength_ = (remainderLength > blockFactor) ? blockFactor :
                       (remainderLength > 0 ? remainderLength : 0);
        blockBaseIndex_ = blockFactor * blockIdx;

        ubLength_ = (tilingData->ubFactor > 0) ? tilingData->ubFactor : 1;
        colNum_ = (tilingData->colNum > 0) ? tilingData->colNum : 1;

        inputGMX.SetGlobalBuffer((__gm__ T*)x + blockBaseIndex_, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + blockBaseIndex_, blockLength_);

        pipe_.InitBuffer(inputQueueX_, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
        pipe_.InitBuffer(outputQueueY_, BUFFER_NUM, static_cast<uint32_t>(ubLength_ * sizeof(T)));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0) {
            return;
        }

        const int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; ++i) {
            const int64_t currentNum =
                (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
            CopyIn(i, currentNum);
            Compute(i, currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        AscendC::LocalTensor<T> xLocal = inputQueueX_.template AllocTensor<T>();

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams,
                            {false, 0, 0, static_cast<T>(0)});
        inputQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t progress, int64_t currentNum)
    {
        AscendC::LocalTensor<T> xLocal = inputQueueX_.template DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outputQueueY_.template AllocTensor<T>();

        const int64_t blockOffset = progress * ubLength_;
        for (int64_t i = 0; i < currentNum; ++i) {
            const int64_t globalIndex = blockBaseIndex_ + blockOffset + i;
            const int64_t row = globalIndex / colNum_;
            const int64_t col = globalIndex - row * colNum_;

            T value = xLocal.GetValue(static_cast<int32_t>(i));
            if (col > row) {
                value = static_cast<T>(0);
            }
            yLocal.SetValue(static_cast<int32_t>(i), value);
        }

        outputQueueY_.EnQue(yLocal);
        inputQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        AscendC::LocalTensor<T> yLocal = outputQueueY_.template DeQue<T>();

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
        outputQueueY_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inputQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueueY_;

    AscendC::GlobalTensor<T> inputGMX;
    AscendC::GlobalTensor<T> outputGMY;

    int64_t blockBaseIndex_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t colNum_ = 0;
};

} // namespace NsTril

#endif // TRIL_ARCH22_H