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
 * \file std.h
 * \brief Std 算子 kernel 类定义（arch22 架构）
 *
 * 本实现针对 float32 二维 C 连续张量，沿最后一维（axis=1）计算总体标准差（ddof=0）。
 * 采用两遍算法保证数值精度：第一遍计算均值，第二遍计算平方偏差和。
 */

#ifndef STD_H
#define STD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "std_tiling_data.h"

namespace NsStd {
using namespace AscendC;

template <typename T, int BUFFER_MODE = 0>
class Std {
    static constexpr int32_t BUFFER_NUM = (BUFFER_MODE == 0) ? 1 : 2;

public:
    __aicore__ inline Std() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const StdTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t row, int64_t colOffset, int64_t currentNum);
    __aicore__ inline float SumTile(int64_t currentNum);
    __aicore__ inline float SumSquaredDiffTile(int64_t currentNum, float mean);
    __aicore__ inline void ComputeRow(int64_t row);
    __aicore__ inline void CopyOut(int64_t row, float value);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue;
    TBuf<QuePosition::VECCALC> sqBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TBuf<QuePosition::VECCALC> sqrtBuf;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t totalRows_ = 0;
    int64_t totalCols_ = 0;
    int64_t blockFactor_ = 0;
    int64_t ubFactor_ = 0;
    int64_t rowStart_ = 0;
    int64_t rowEnd_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Std<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, const StdTilingData* tilingData)
{
    totalRows_ = tilingData->totalRows;
    totalCols_ = tilingData->totalCols;
    blockFactor_ = tilingData->blockFactor;
    ubFactor_ = tilingData->ubFactor;

    rowStart_ = AscendC::GetBlockIdx() * blockFactor_;
    if (rowStart_ > totalRows_) {
        rowStart_ = totalRows_;
    }
    rowEnd_ = rowStart_ + blockFactor_;
    if (rowEnd_ > totalRows_) {
        rowEnd_ = totalRows_;
    }

    inputGM.SetGlobalBuffer((__gm__ T*)x, totalRows_ * totalCols_);
    outputGM.SetGlobalBuffer((__gm__ T*)y, totalRows_);

    pipe.InitBuffer(inputQue, BUFFER_NUM, ubFactor_ * sizeof(T));
    pipe.InitBuffer(sqBuf, ubFactor_ * sizeof(T));
    pipe.InitBuffer(workBuf, ubFactor_ * sizeof(T));
    pipe.InitBuffer(reduceBuf, 64 * sizeof(T));
    pipe.InitBuffer(sqrtBuf, 64 * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Std<T, BUFFER_MODE>::CopyIn(int64_t row, int64_t colOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQue.AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    int64_t srcOffset = row * totalCols_ + colOffset;
    DataCopyPad(xLocal, inputGM[srcOffset], copyParams, {false, 0, 0, static_cast<T>(0)});
    inputQue.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline float Std<T, BUFFER_MODE>::SumTile(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQue.DeQue<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> reduceLocal = reduceBuf.Get<T>();
    ReduceSum<T>(reduceLocal, xLocal, workLocal, currentNum);
    float val = reduceLocal.GetValue(0);
    inputQue.FreeTensor(xLocal);
    return val;
}

template <typename T, int BUFFER_MODE>
__aicore__ inline float Std<T, BUFFER_MODE>::SumSquaredDiffTile(int64_t currentNum, float mean)
{
    LocalTensor<T> xLocal = inputQue.DeQue<T>();
    LocalTensor<T> sqLocal = sqBuf.Get<T>();
    LocalTensor<T> workLocal = workBuf.Get<T>();
    LocalTensor<T> reduceLocal = reduceBuf.Get<T>();
    Adds(xLocal, xLocal, static_cast<T>(-mean), currentNum);
    Mul(sqLocal, xLocal, xLocal, currentNum);
    ReduceSum<T>(reduceLocal, sqLocal, workLocal, currentNum);
    float val = reduceLocal.GetValue(0);
    inputQue.FreeTensor(xLocal);
    return val;
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Std<T, BUFFER_MODE>::ComputeRow(int64_t row)
{
    float sum = 0.0f;
    int64_t colOffset = 0;
    while (colOffset < totalCols_) {
        int64_t currentNum = ubFactor_;
        if (colOffset + currentNum > totalCols_) {
            currentNum = totalCols_ - colOffset;
        }
        CopyIn(row, colOffset, currentNum);
        sum += SumTile(currentNum);
        colOffset += currentNum;
    }

    float mean = sum / static_cast<float>(totalCols_);

    float sumsq = 0.0f;
    colOffset = 0;
    while (colOffset < totalCols_) {
        int64_t currentNum = ubFactor_;
        if (colOffset + currentNum > totalCols_) {
            currentNum = totalCols_ - colOffset;
        }
        CopyIn(row, colOffset, currentNum);
        sumsq += SumSquaredDiffTile(currentNum, mean);
        colOffset += currentNum;
    }

    float var = sumsq / static_cast<float>(totalCols_);
    if (var < 0.0f) {
        var = 0.0f;
    }

    LocalTensor<T> sqrtLocal = sqrtBuf.Get<T>();
    Duplicate(sqrtLocal, static_cast<T>(var), 64);
    Sqrt(sqrtLocal, sqrtLocal, 64);
    float stdVal = sqrtLocal.GetValue(0);
    CopyOut(row, stdVal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Std<T, BUFFER_MODE>::CopyOut(int64_t row, float value)
{
    outputGM.SetValue(static_cast<uint32_t>(row), static_cast<T>(value));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Std<T, BUFFER_MODE>::Process()
{
    for (int64_t row = rowStart_; row < rowEnd_; ++row) {
        ComputeRow(row);
    }
}

} // namespace NsStd

#endif // STD_H