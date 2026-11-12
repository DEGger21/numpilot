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
    __aicore__ inline float ComputeRowVar(int64_t row);
    __aicore__ inline void WriteRow(int64_t row, float var);

private:
    // One float32 result per 32-byte UB block: vector instructions require
    // 32-byte-aligned UB addresses, so a packed 4-byte slot would fault.
    static constexpr int64_t OUT_BLOCK_ELEMS = 8;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue;
    TBuf<QuePosition::VECCALC> sqBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TQue<QuePosition::VECOUT, 1> outQue;

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

    // UB blocks are allocated in 32-byte units; a narrow reduction (cols=4)
    // would otherwise request a 16-byte tile, which is below the minimum.
    int64_t tileBytes = ubFactor_ * static_cast<int64_t>(sizeof(T));
    tileBytes = (tileBytes + 31) / 32 * 32;

    pipe.InitBuffer(inputQue, BUFFER_NUM, tileBytes);
    pipe.InitBuffer(sqBuf, tileBytes);
    pipe.InitBuffer(workBuf, tileBytes);
    pipe.InitBuffer(reduceBuf, 64 * sizeof(T));
    pipe.InitBuffer(outQue, 1, OUT_BLOCK_ELEMS * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Std<T, BUFFER_MODE>::CopyIn(int64_t row, int64_t colOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQue.template AllocTensor<T>();
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
    LocalTensor<T> xLocal = inputQue.template DeQue<T>();
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
    LocalTensor<T> xLocal = inputQue.template DeQue<T>();
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
__aicore__ inline float Std<T, BUFFER_MODE>::ComputeRowVar(int64_t row)
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

    // Return the variance; clamp (var < 0) and Sqrt happen on the whole output
    // tile in FlushChunk so the negative-noise guard stays vector-friendly.
    return sumsq / static_cast<float>(totalCols_);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Std<T, BUFFER_MODE>::WriteRow(int64_t row, float var)
{
    LocalTensor<T> outLocal = outQue.template AllocTensor<T>();
    Duplicate(outLocal, static_cast<T>(var), static_cast<int32_t>(OUT_BLOCK_ELEMS));
    Maxs(outLocal, outLocal, static_cast<T>(0), static_cast<int32_t>(OUT_BLOCK_ELEMS));
    Sqrt(outLocal, outLocal, static_cast<int32_t>(OUT_BLOCK_ELEMS));

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    // The VECOUT queue owns both hand-offs: EnQue/DeQue order the vector
    // writes before the MTE3 copy, FreeTensor orders the copy before the
    // block is handed out again.
    outQue.EnQue(outLocal);
    LocalTensor<T> readyLocal = outQue.template DeQue<T>();
    DataCopyPad(outputGM[row], readyLocal, copyParams);
    outQue.FreeTensor(readyLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Std<T, BUFFER_MODE>::Process()
{
    if (rowStart_ >= rowEnd_) {
        return;
    }

    // Scalar stores to GM are cached per AI core and were observed to be
    // invisible to the subsequent D2H copy (a random subset of rows stayed at
    // the caller's pre-fill).  Results therefore leave the core through an
    // MTE3 DataCopyPad, which is cache-coherent.
    for (int64_t row = rowStart_; row < rowEnd_; ++row) {
        WriteRow(row, ComputeRowVar(row));
    }
}

} // namespace NsStd

#endif // STD_H