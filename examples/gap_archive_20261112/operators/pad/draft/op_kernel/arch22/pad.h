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
 * \brief Pad (constant mode) kernel class for arch22 / Ascend910B3.
 *
 * Bounded support scope for this candidate:
 *   - dtype: float32 only (no implicit casts from other dtypes)
 *   - rank: 2 (C-contiguous input and output)
 *   - mode: "constant" with NumPy's default constant_values = 0
 *   - pads: non-negative top/bottom/left/right values
 * Anything outside this scope is rejected by the host adapter / ACLNN layer
 * before the kernel is launched.
 *
 * Tiling data contract (op_kernel/arch22/pad_tiling_data.h):
 *   struct PadTilingData {
 *       int64_t inH;     // input height (rows)
 *       int64_t inW;     // input width (columns)
 *       int64_t outH;    // inH + padTop + padBottom
 *       int64_t outW;    // inW + padLeft + padRight
 *       int64_t padTop;  // rows added above the input, >= 0
 *       int64_t padLeft; // columns added to the left of the input, >= 0
 *   };
 * Additional host-side bookkeeping fields may be present; the kernel only
 * depends on the six fields above.
 *
 * Implementation notes:
 *   - Output rows are partitioned across AI cores by row, using GetBlockNum()
 *     and GetBlockIdx(). Any partition is numerically valid.
 *   - Output row starts are not guaranteed to be 32-byte aligned (the observed
 *     output width is 2052 float32 elements), therefore byte-granular
 *     DataCopyPad transfers are used rather than alignment-constrained
 *     DataCopy transfers.
 *   - Each tile follows the required Init -> CopyIn -> Compute -> CopyOut
 *     pipeline. CopyIn is a no-op for pure padding tiles; Compute is a no-op
 *     for tiles whose payload is an input row segment.
 */

#ifndef PAD_KERNEL_ARCH22_H
#define PAD_KERNEL_ARCH22_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "pad_tiling_data.h"
#include "pad_tiling_key.h"

namespace NsPad {

using namespace AscendC;

template <typename T>
class Pad {
public:
    __aicore__ inline Pad() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PadTilingData* tilingData)
    {
        inH_ = tilingData->inH;
        inW_ = tilingData->inW;
        outH_ = tilingData->outH;
        outW_ = tilingData->outW;
        padTop_ = tilingData->padTop;
        padLeft_ = tilingData->padLeft;

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_.InitBuffer(inputQueue_, 1, static_cast<uint32_t>(kUbCols * sizeof(T)));
        pipe_.InitBuffer(outputQueue_, 1, static_cast<uint32_t>(kUbCols * sizeof(T)));
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
    // Upper bound on the number of float32 elements staged through UB per tile.
    // 8192 elements = 32 KiB per queue; two queues use 64 KiB of the 192 KiB UB
    // available on Ascend910B3.
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

    __aicore__ inline void FillPad(int64_t outOffset, int64_t count)
    {
        int64_t done = 0;
        while (done < count) {
            int64_t len = count - done;
            if (len > kUbCols) {
                len = kUbCols;
            }
            ProcessChunk(0, outOffset + done, len, false);
            done += len;
        }
    }

    __aicore__ inline void CopyInput(int64_t inOffset, int64_t outOffset, int64_t count)
    {
        int64_t done = 0;
        while (done < count) {
            int64_t len = count - done;
            if (len > kUbCols) {
                len = kUbCols;
            }
            ProcessChunk(inOffset + done, outOffset + done, len, true);
            done += len;
        }
    }

    __aicore__ inline void ProcessChunk(int64_t inOffset, int64_t outOffset, int64_t count, bool useInput)
    {
        CopyIn(inOffset, count, useInput);
        Compute(count, useInput);
        CopyOut(outOffset, count, useInput);
    }

    __aicore__ inline void CopyIn(int64_t inOffset, int64_t count, bool useInput)
    {
        if (!useInput) {
            return;
        }

        LocalTensor<T> local = inputQueue_.template AllocTensor<T>();

        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;

        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(local, inputGm_[inOffset], params, padParams);

        inputQueue_.EnQue(local);
    }

    __aicore__ inline void Compute(int64_t count, bool useInput)
    {
        if (useInput) {
            return;
        }

        LocalTensor<T> local = outputQueue_.template AllocTensor<T>();
        AscendC::Duplicate(local, static_cast<T>(0), static_cast<int32_t>(count));
        outputQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(int64_t outOffset, int64_t count, bool useInput)
    {
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;

        if (useInput) {
            LocalTensor<T> local = inputQueue_.template DeQue<T>();
            AscendC::DataCopyPad(outputGm_[outOffset], local, params);
            inputQueue_.FreeTensor(local);
        } else {
            LocalTensor<T> local = outputQueue_.template DeQue<T>();
            AscendC::DataCopyPad(outputGm_[outOffset], local, params);
            outputQueue_.FreeTensor(local);
        }
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
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

#endif  // PAD_KERNEL_ARCH22_H