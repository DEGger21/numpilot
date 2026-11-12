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
 * \file pad.h
 * \brief Pad (constant mode) kernel class for arch22 / AscendB3.
 *
 * Bounded support scope for this candidate:
 *   - dtype: float32 only (no implicit casts from other dtypes)
 *   - rank: 2 (C-contiguous input and output)
 *   - mode: "constant" with NumPy's default constant_values = 0
 *   - pads: non-negative top/bottom/left/right values
 * Anything outside this scope is rejected by the host adapter / ACLNN layer
 * before the kernel is launched.
 *
 * Transfer discipline (cannbot `ascendc-api-best-practices`, api-datacopy):
 *   - DataCopyPad for every GM <-> UB transfer; blockLen stays the exact byte
 *     length of the region.  CopyOut drops the alignment dummy on the GM side,
 *     so a boundary segment shorter than 32 B is transferred exactly.
 *   - Every transfer stages through UB element offset 0, so each DataCopyPad
 *     UB address is 32-byte aligned (hard MTE requirement; a non-aligned UB
 *     start raised "The access address of the MTE instruction is not aligned
 *     with the data type bit width" in v8).
 *
 * Synchronisation note: the input segment is copied GM -> UB -> GM with no
 * vector op in between.  `TQue<QuePosition::VECIN>` EnQue/DeQue orders MTE2
 * against the *vector* pipe, not against MTE3, so the direct CopyIn->CopyOut
 * pair in v7/v10 raced: output rows occasionally carried a neighbouring row's
 * data or the partial-buffer dummy fill.  An explicit
 * `PipeBarrier<PIPE_ALL>()` between the two transfers serialises them.
 *
 * The tiling struct is read straight from its GM buffer.  The macro variant
 * used by v7 staged it through UB address 0, which is also the first queue
 * base.
 */

#ifndef PAD_KERNEL_ARCH22_H
#define PAD_KERNEL_ARCH22_H

#include "kernel_operator.h"
#include "pad_tiling_data.h"

namespace NsPad {

using namespace AscendC;

template <typename T>
class Pad {
public:
    __aicore__ inline Pad() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const __gm__ PadTilingData* tilingData)
    {
        inH_ = tilingData->inH;
        inW_ = tilingData->inW;
        outH_ = tilingData->outH;
        outW_ = tilingData->outW;
        padTop_ = tilingData->padTop;
        padLeft_ = tilingData->padLeft;

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_.InitBuffer(rowBuffer_, static_cast<uint32_t>(kUbCols * sizeof(T)));
    }

    __aicore__ inline void Process()
    {
        if (outH_ <= 0 || outW_ <= 0) {
            return;
        }

        int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        if (blockNum <= 0) {
            blockNum = 1;
        }
        const int64_t rowsPerCore = (outH_ + blockNum - 1) / blockNum;
        if (rowsPerCore <= 0) {
            return;
        }

        const int64_t rowStart = static_cast<int64_t>(AscendC::GetBlockIdx()) * rowsPerCore;
        if (rowStart >= outH_) {
            return;
        }
        int64_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > outH_) {
            rowEnd = outH_;
        }

        for (int64_t outRow = rowStart; outRow < rowEnd; ++outRow) {
            ProcessRow(outRow);
        }
    }

private:
    // Elements staged per transfer: 8192 float32 = 32 KiB.
    static constexpr int64_t kUbCols = 8192;

    __aicore__ inline void ProcessRow(int64_t outRow)
    {
        const int64_t outBase = outRow * outW_;
        const int64_t inRow = outRow - padTop_;

        if (inRow < 0 || inRow >= inH_) {
            FillPad(outBase, outW_);
            return;
        }

        const int64_t inBase = inRow * inW_;
        if (padLeft_ > 0) {
            FillPad(outBase, padLeft_);
        }
        if (inW_ > 0) {
            CopyInput(inBase, outBase + padLeft_, inW_);
        }

        const int64_t rightWidth = outW_ - padLeft_ - inW_;
        if (rightWidth > 0) {
            FillPad(outBase + padLeft_ + inW_, rightWidth);
        }
    }

    // Write `count` zeros to outputGm_ starting at `outOffset`.
    __aicore__ inline void FillPad(int64_t outOffset, int64_t count)
    {
        int64_t done = 0;
        while (done < count) {
            int64_t len = count - done;
            if (len > kUbCols) {
                len = kUbCols;
            }

            LocalTensor<T> local = rowBuffer_.Get<T>();
            AscendC::Duplicate(local, static_cast<T>(0), static_cast<int32_t>(len));
            AscendC::PipeBarrier<PIPE_ALL>();
            CopyOut(outOffset + done, len);
            done += len;
        }
    }

    // Copy `count` elements of inputGm_ starting at `inOffset` to outputGm_
    // starting at `outOffset`.
    __aicore__ inline void CopyInput(int64_t inOffset, int64_t outOffset, int64_t count)
    {
        int64_t done = 0;
        while (done < count) {
            int64_t len = count - done;
            if (len > kUbCols) {
                len = kUbCols;
            }

            LocalTensor<T> local = rowBuffer_.Get<T>();
            AscendC::DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(len * sizeof(T));
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;

            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(local, inputGm_[inOffset + done], params, padParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            CopyOut(outOffset + done, len);
            done += len;
        }
    }

    __aicore__ inline void CopyOut(int64_t outOffset, int64_t count)
    {
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;

        LocalTensor<T> local = rowBuffer_.Get<T>();
        AscendC::DataCopyPad(outputGm_[outOffset], local, params);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    int64_t inH_ = 0;
    int64_t inW_ = 0;
    int64_t outH_ = 0;
    int64_t outW_ = 0;
    int64_t padTop_ = 0;
    int64_t padLeft_ = 0;
};

}  // namespace NsPad

#endif // PAD_KERNEL_ARCH22_H
