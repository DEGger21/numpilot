/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_gradient_descent.h
 * \brief ApplyGradientDescent 算子 kernel 类定义（arch22 架构）
 *
 * 语义（NumPy oracle）：out = var - alpha * delta，非原地，仅 float32。
 * 流水线：CopyIn -> Compute -> CopyOut。
 */
#ifndef APPLY_GRADIENT_DESCENT_H
#define APPLY_GRADIENT_DESCENT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_gradient_descent_tiling_data.h"
#include "apply_gradient_descent_tiling_key.h"

namespace NsApplyGradientDescent {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class ApplyGradientDescent {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline ApplyGradientDescent(){};

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR alpha, GM_ADDR delta, GM_ADDR output,
                                const ApplyGradientDescentTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueueVar;
    TQue<QuePosition::VECIN, 1> inputQueueAlpha;
    TQue<QuePosition::VECIN, 1> inputQueueDelta;
    TQue<QuePosition::VECOUT, 1> outputQueueOut;

    GlobalTensor<T> inputGMVar;
    GlobalTensor<T> inputGMAlpha;
    GlobalTensor<T> inputGMDelta;
    GlobalTensor<T> outputGMOut;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyGradientDescent<T, BUFFER_MODE>::Init(GM_ADDR var, GM_ADDR alpha, GM_ADDR delta,
                                                                  GM_ADDR output,
                                                                  const ApplyGradientDescentTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMVar.SetGlobalBuffer((__gm__ T*)var + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    inputGMAlpha.SetGlobalBuffer((__gm__ T*)alpha, 1);
    inputGMDelta.SetGlobalBuffer((__gm__ T*)delta + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMOut.SetGlobalBuffer((__gm__ T*)output + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueVar, BUFFER_NUM, ubLength_ * sizeof(T));
    // alpha 仅一个元素，UB 分配至少一个 32 字节块
    pipe.InitBuffer(inputQueueAlpha, BUFFER_NUM, 32);
    pipe.InitBuffer(inputQueueDelta, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueOut, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyGradientDescent<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> varLocal = inputQueueVar.template AllocTensor<T>();
    AscendC::LocalTensor<T> alphaLocal = inputQueueAlpha.template AllocTensor<T>();
    AscendC::LocalTensor<T> deltaLocal = inputQueueDelta.template AllocTensor<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(varLocal, inputGMVar[progress * ubLength_], copyParams, {false, 0, 0, static_cast<T>(0)});
    AscendC::DataCopyPad(deltaLocal, inputGMDelta[progress * ubLength_], copyParams, {false, 0, 0, static_cast<T>(0)});
    AscendC::DataCopyExtParams alphaCopyParams;
    alphaCopyParams.blockCount = 1;
    alphaCopyParams.blockLen = sizeof(T);
    alphaCopyParams.srcStride = 0;
    alphaCopyParams.dstStride = 0;
    alphaCopyParams.rsv = 0;
    AscendC::DataCopyPad(alphaLocal, inputGMAlpha, alphaCopyParams, {false, 0, 0, static_cast<T>(0)});
    inputQueueVar.EnQue(varLocal);
    inputQueueAlpha.EnQue(alphaLocal);
    inputQueueDelta.EnQue(deltaLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyGradientDescent<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> outLocal = outputQueueOut.template DeQue<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(outputGMOut[progress * ubLength_], outLocal, copyParams);
    outputQueueOut.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyGradientDescent<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> varLocal = inputQueueVar.template DeQue<T>();
    AscendC::LocalTensor<T> alphaLocal = inputQueueAlpha.template DeQue<T>();
    AscendC::LocalTensor<T> deltaLocal = inputQueueDelta.template DeQue<T>();
    AscendC::LocalTensor<T> outLocal = outputQueueOut.template AllocTensor<T>();
    // out = var - alpha * delta（逐元素，alpha 为标量广播）
    T alphaValue = alphaLocal.GetValue(0);
    AscendC::Muls(outLocal, deltaLocal, alphaValue, currentNum);
    AscendC::Sub(outLocal, varLocal, outLocal, currentNum);
    outputQueueOut.template EnQue<T>(outLocal);
    inputQueueVar.FreeTensor(varLocal);
    inputQueueAlpha.FreeTensor(alphaLocal);
    inputQueueDelta.FreeTensor(deltaLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyGradientDescent<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsApplyGradientDescent
#endif // APPLY_GRADIENT_DESCENT_H