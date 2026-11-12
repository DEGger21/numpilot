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
 * \file ediff1d.h
 * \brief Ediff1d 算子 kernel 类定义（arch22 架构）
 *
 * 语义：对输入按 C 连续顺序展平后计算相邻元素差分。
 * 输入元素个数为 N，输出元素个数为 N - 1。
 *
 * TilingData 约定：
 *   - totalNum:    输出元素总数 N - 1（与 Ediff1dTilingData 的定义一致）
 *   - blockFactor: 每个核负责的输出元素个数
 *   - ubFactor:    每次 UB 循环处理的输出元素个数
 *
 * 由于相邻差分需要 x[i] 与 x[i + 1]，CopyIn 阶段分别搬运
 * 起始于当前输出块偏移和偏移 + 1 的两段输入到两个对齐的 UB
 * 缓冲区，再执行向量减法，避免非对齐 UB 向量操作数。
 */

#ifndef EDIFF1D_H
#define EDIFF1D_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ediff1d_tiling_data.h"

namespace NsEdiff1d {

using namespace AscendC;

template <typename T = float, int BUFFER_MODE = 0>
class Ediff1d {
    static_assert(std::is_same<T, float>::value,
                  "Ediff1d draft supports float32 input only; other dtypes must be rejected explicitly.");
    static constexpr int32_t BUFFER_NUM = (BUFFER_MODE == 0) ? 1 : 2;

public:
    __aicore__ inline Ediff1d() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Ediff1dTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Ediff1d<T, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, const Ediff1dTilingData* tilingData)
{
    int64_t coreIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    // Ediff1dTilingData.totalNum is already the output count (N - 1).
    int64_t outputNum = (tilingData->totalNum > 0) ? tilingData->totalNum : 0;
    int64_t start = tilingData->blockFactor * coreIdx;
    int64_t remainder = outputNum - start;

    if (remainder <= 0) {
        blockLength_ = 0;
    } else {
        blockLength_ = (remainder > tilingData->blockFactor) ? tilingData->blockFactor : remainder;
    }

    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = blockLength_;
    }

    int64_t inputLength = (blockLength_ > 0) ? (blockLength_ + 1) : 1;
    int64_t outputLength = (blockLength_ > 0) ? blockLength_ : 1;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + start, inputLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + start, outputLength);

    if (blockLength_ > 0 && ubLength_ > 0) {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(inputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueZ, BUFFER_NUM, ubLength_ * sizeof(T));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Ediff1d<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.template AllocTensor<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams,
                         {false, 0, 0, static_cast<T>(0)});
    AscendC::DataCopyPad(yLocal, inputGMX[progress * ubLength_ + 1], copyParams,
                         {false, 0, 0, static_cast<T>(0)});

    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Ediff1d<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.template DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.template AllocTensor<T>();

    AscendC::Sub(zLocal, yLocal, xLocal, static_cast<int32_t>(currentNum));

    outputQueueZ.template EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Ediff1d<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.template DeQue<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    AscendC::DataCopyPad(outputGMY[progress * ubLength_], zLocal, copyParams);
    outputQueueZ.FreeTensor(zLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Ediff1d<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
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

} // namespace NsEdiff1d

#endif // EDIFF1D_H