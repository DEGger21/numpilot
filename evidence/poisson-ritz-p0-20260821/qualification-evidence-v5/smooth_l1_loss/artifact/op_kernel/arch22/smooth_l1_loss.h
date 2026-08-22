/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss.h
 * \brief SmoothL1Loss 算子 kernel 类定义（arch22 架构）
 *
 * 语义（NumPy oracle，mean / beta=1）：
 *   error     = predict - label
 *   absolute  = |error|
 *   bounded   = min(absolute, 1.0)
 *   quadratic = bounded * bounded / 2.0
 *   linear    = absolute - bounded
 *   output    = mean(quadratic + linear)   // 标量输出
 *
 * 实现方式：每个核分块计算元素级 loss，块内 ReduceSum 得到局部和，
 * 乘以 1/totalNum 后通过 atomic add 累加到全局标量输出上。
 * 输出缓冲区由 host 侧（L0 API）预先清零。
 */

#ifndef SMOOTH_L1_LOSS_H
#define SMOOTH_L1_LOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "smooth_l1_loss_tiling_data.h"
#include "smooth_l1_loss_tiling_key.h"

namespace NsSmoothL1Loss {

using namespace AscendC;

template <typename T>
class SmoothL1Loss {
    static constexpr int32_t BUFFER_NUM = 2;

public:
    __aicore__ inline SmoothL1Loss(){};

    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR output,
                                const SmoothL1LossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> predictQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> labelQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> lossQueue_;

    GlobalTensor<T> predictGM_;
    GlobalTensor<T> labelGM_;
    GlobalTensor<T> outputGM_;

    TBuf<TPosition::VECCALC> errorBuf_;
    TBuf<TPosition::VECCALC> absBuf_;
    TBuf<TPosition::VECCALC> boundedBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;

    int64_t blockLength_ = 0;  // 本核处理的元素个数
    int64_t ubLength_ = 0;     // 每次 UB 循环处理的元素个数
    int64_t totalNum_ = 0;     // 全部元素个数（mean 归一化用）
    float blockSum_ = 0.0F;    // 本核累加和
};

template <typename T>
__aicore__ inline void SmoothL1Loss<T>::Init(GM_ADDR predict, GM_ADDR label, GM_ADDR output,
                                             const SmoothL1LossTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    int64_t remainderLength = totalNum_ - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;
    blockSum_ = 0.0F;

    if (blockLength_ <= 0) {
        // 本核无数据，仍需初始化队列以保证 Process 安全返回
        predictGM_.SetGlobalBuffer((__gm__ T*)predict, 0);
        labelGM_.SetGlobalBuffer((__gm__ T*)label, 0);
        outputGM_.SetGlobalBuffer((__gm__ T*)output, 0);
        pipe.InitBuffer(predictQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(labelQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(lossQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
        return;
    }

    predictGM_.SetGlobalBuffer((__gm__ T*)predict + tilingData->blockFactor * AscendC::GetBlockIdx(),
                               blockLength_);
    labelGM_.SetGlobalBuffer((__gm__ T*)label + tilingData->blockFactor * AscendC::GetBlockIdx(),
                             blockLength_);
    outputGM_.SetGlobalBuffer((__gm__ T*)output, 1);

    pipe.InitBuffer(predictQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(labelQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(lossQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(errorBuf_, ubLength_ * sizeof(T));
    pipe.InitBuffer(absBuf_, ubLength_ * sizeof(T));
    pipe.InitBuffer(boundedBuf_, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf_, ubLength_ * sizeof(T));
    pipe.InitBuffer(sumBuf_, 32); // ReduceSum 输出占位，32 字节对齐
}

template <typename T>
__aicore__ inline void SmoothL1Loss<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> predictLocal = predictQueue_.template AllocTensor<T>();
    AscendC::LocalTensor<T> labelLocal = labelQueue_.template AllocTensor<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(predictLocal, predictGM_[progress * ubLength_], copyParams, {false, 0, 0, static_cast<T>(0)});
    AscendC::DataCopyPad(labelLocal, labelGM_[progress * ubLength_], copyParams, {false, 0, 0, static_cast<T>(0)});
    predictQueue_.EnQue(predictLocal);
    labelQueue_.EnQue(labelLocal);
}

template <typename T>
__aicore__ inline void SmoothL1Loss<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> predictLocal = predictQueue_.template DeQue<T>();
    AscendC::LocalTensor<T> labelLocal = labelQueue_.template DeQue<T>();
    AscendC::LocalTensor<T> lossLocal = lossQueue_.template AllocTensor<T>();

    AscendC::LocalTensor<T> errorLocal = errorBuf_.template Get<T>();
    AscendC::LocalTensor<T> absLocal = absBuf_.template Get<T>();
    AscendC::LocalTensor<T> boundedLocal = boundedBuf_.template Get<T>();
    AscendC::LocalTensor<T> tmpLocal = tmpBuf_.template Get<T>();

    // error = predict - label; absolute = |error|
    AscendC::Sub(errorLocal, predictLocal, labelLocal, currentNum);
    AscendC::Abs(absLocal, errorLocal, currentNum);

    // bounded = min(absolute, 1.0)
    AscendC::Mins(boundedLocal, absLocal, static_cast<T>(1.0), currentNum);

    // quadratic = bounded * bounded / 2.0
    AscendC::Mul(tmpLocal, boundedLocal, boundedLocal, currentNum);
    AscendC::Muls(tmpLocal, tmpLocal, static_cast<T>(0.5), currentNum);

    // linear = absolute - bounded
    AscendC::Sub(lossLocal, absLocal, boundedLocal, currentNum);

    // loss = quadratic + linear
    AscendC::Add(lossLocal, lossLocal, tmpLocal, currentNum);

    // 块内求和，累加到本核局部和
    AscendC::LocalTensor<T> sumLocal = sumBuf_.template Get<T>();
    AscendC::ReduceSum(sumLocal, lossLocal, tmpLocal, currentNum);
    blockSum_ += static_cast<float>(sumLocal.GetValue(0));

    lossQueue_.template EnQue<T>(lossLocal);
    predictQueue_.FreeTensor(predictLocal);
    labelQueue_.FreeTensor(labelLocal);
}

template <typename T>
__aicore__ inline void SmoothL1Loss<T>::CopyOut()
{
    // 归一化 mean：本核累加和 / totalNum，atomic add 到全局标量输出
    AscendC::LocalTensor<T> sumLocal = sumBuf_.template Get<T>();
    sumLocal.SetValue(0, static_cast<T>(blockSum_ / static_cast<float>(totalNum_)));

    AscendC::SetAtomicAdd<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(outputGM_, sumLocal, copyParams);
    AscendC::SetAtomicNone();
}

template <typename T>
__aicore__ inline void SmoothL1Loss<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
    }
    // lossQueue_ 中的中间结果无需写回 GM，仅释放
    CopyOut();
}

} // namespace NsSmoothL1Loss
#endif // SMOOTH_L1_LOSS_H