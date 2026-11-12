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
 * \brief Tril kernel class for arch22 / Ascend910B3.
 *
 * NumPy semantics: y[i, j] = x[i, j] for j <= i, 0 otherwise (k = 0).
 * Admitted profile: one float32 rank-2 C-contiguous tensor; other dtypes and
 * ranks are rejected by the host before launch.
 *
 * Design (row-parallel, vectorised):
 *   - Rows are split across AI cores; each core walks its rows.
 *   - A row is streamed through UB in segments of at most kUbCols elements.
 *     Within a segment, columns that survive the mask are copied GM -> UB ->
 *     GM with DataCopyPad; the masked tail is produced with Duplicate (zero
 *     fill) in UB and copied out.  No scalar GetValue/SetValue access is used
 *     (blacklisted for production code by the cannbot
 *     `ascendc-api-best-practices` skill).  The generated draft looped over
 *     every element with GetValue/SetValue; this replaces it.
 *
 * Transfer discipline (cannbot `ascendc-api-best-practices`, api-datacopy):
 *   - DataCopyPad for every GM <-> UB transfer, blockLen the exact byte length
 *     of the region.  CopyOut discards the 32 B alignment dummy on the GM side,
 *     so boundary segments shorter than 32 B are transferred exactly.
 *   - Every transfer stages through UB element offset 0, so each DataCopyPad UB
 *     address is 32-byte aligned (hard MTE requirement).
 *   - A TQue VECIN EnQue/DeQue orders MTE2 against the vector pipe, not MTE3,
 *     so an adjacent CopyIn -> CopyOut pair can race (observed in the pad
 *     kernel).  This kernel uses a TBuf staging buffer with explicit
 *     PipeBarrier<PIPE_ALL>() after every transfer and after Duplicate.
 *
 * The tiling struct is read straight from its GM buffer (the macro variant
 * stages it through UB address 0, which is the staging buffer base).
 */

#ifndef TRIL_ARCH22_H
#define TRIL_ARCH22_H

#include "kernel_operator.h"
#include "tril_tiling_data.h"

namespace NsTril {

using namespace AscendC;

template <typename T>
class Tril {
public:
    __aicore__ inline Tril() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const __gm__ TrilTilingData* tilingData)
    {
        rows_ = tilingData->rows;
        cols_ = tilingData->cols;

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_.InitBuffer(rowBuffer_, static_cast<uint32_t>(kUbCols * sizeof(T)));
    }

    __aicore__ inline void Process()
    {
        if (rows_ <= 0 || cols_ <= 0) {
            return;
        }

        int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        if (blockNum <= 0) {
            blockNum = 1;
        }
        const int64_t rowsPerCore = (rows_ + blockNum - 1) / blockNum;
        if (rowsPerCore <= 0) {
            return;
        }

        const int64_t rowStart = static_cast<int64_t>(AscendC::GetBlockIdx()) * rowsPerCore;
        if (rowStart >= rows_) {
            return;
        }
        int64_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > rows_) {
            rowEnd = rows_;
        }

        for (int64_t row = rowStart; row < rowEnd; ++row) {
            ProcessRow(row);
        }
    }

private:
    // Elements staged per transfer: 8192 float32 = 32 KiB.
    static constexpr int64_t kUbCols = 8192;

    __aicore__ inline void ProcessRow(int64_t row)
    {
        const int64_t base = row * cols_;
        // Columns [0, keep) are kept from the input; [keep, cols) are zero.
        const int64_t keep = (row + 1 < cols_) ? (row + 1) : cols_;

        int64_t done = 0;
        while (done < cols_) {
            int64_t len = cols_ - done;
            if (len > kUbCols) {
                len = kUbCols;
            }

            int64_t keepInSeg = keep - done;
            if (keepInSeg < 0) {
                keepInSeg = 0;
            }
            if (keepInSeg > len) {
                keepInSeg = len;
            }

            if (keepInSeg > 0) {
                // Copy in only the surviving prefix, then write it back out.
                CopyIn(base + done, keepInSeg);
                CopyOut(base + done, keepInSeg);
            }
            if (keepInSeg < len) {
                const int64_t tail = len - keepInSeg;
                FillZero(tail);
                CopyOut(base + done + keepInSeg, tail);
            }

            done += len;
        }
    }

    // GM -> UB (element offset 0), exact `count` elements.
    __aicore__ inline void CopyIn(int64_t inOffset, int64_t count)
    {
        AscendC::LocalTensor<T> local = rowBuffer_.Get<T>();

        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;

        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(local, inputGm_[inOffset], params, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Zero the first `count` elements of the staging buffer.
    __aicore__ inline void FillZero(int64_t count)
    {
        AscendC::LocalTensor<T> local = rowBuffer_.Get<T>();
        AscendC::Duplicate(local, static_cast<T>(0), static_cast<int32_t>(count));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // UB (element offset 0) -> GM, exact `count` elements.
    __aicore__ inline void CopyOut(int64_t outOffset, int64_t count)
    {
        AscendC::LocalTensor<T> local = rowBuffer_.Get<T>();

        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;

        AscendC::DataCopyPad(outputGm_[outOffset], local, params);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    int64_t rows_ = 0;
    int64_t cols_ = 0;
};

}  // namespace NsTril

#endif // TRIL_ARCH22_H
