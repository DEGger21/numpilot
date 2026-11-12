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
 * \file flip.h
 * \brief Flip kernel class definition for arch22 (Ascend910B3).
 *
 * NumPilot repair of the generated draft.  Three changes relative to the draft:
 *   - the shape comes from FlipTilingData (rows/cols/blockFactor/ubFactor)
 *     instead of the hard-coded 2048 x 2048 constants, so every rank-2 float32
 *     profile the host admits is computed, not just the observed one;
 *   - rows are distributed with the documented "blockFactor = rows per core"
 *     meaning;
 *   - the in-row reversal is one vector Gather per tile instead of a scalar
 *     GetValue/SetValue loop (the scalar loop both ignored the tiling data and
 *     read UB through the scalar pipe without an MTE2 -> S hand-off).
 *
 * The reversal y[c0 + k] = x[cols - 1 - (c0 + k)] is expressed as: copy the
 * mirrored source tile into UB, then gather it with precomputed byte offsets
 * `(n - 1 - k) * sizeof(T)`.  Offsets are built once per core in Init; the
 * vgather offset unit on this architecture is bytes (the built-in
 * upsample_nearest kernel scales element offsets by the element size the same
 * way).
 */

#ifndef FLIP_H
#define FLIP_H

#include "kernel_operator.h"

namespace NsFlip {

using namespace AscendC;

template <typename T, int BUFFER_MODE = 0>
class Flip {
public:
    static constexpr int32_t BUFFER_NUM = (BUFFER_MODE == 0) ? 1 : 2;

    __aicore__ inline Flip() = default;

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingT* tilingData)
    {
        rows_ = tilingData->rows;
        cols_ = tilingData->cols;
        const int64_t totalNum = rows_ * cols_;
        inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalNum);
        outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalNum);

        int64_t ubFactor = tilingData->ubFactor;
        if (ubFactor <= 0) {
            ubFactor = (cols_ > 0) ? cols_ : 1;
        }
        tileElems_ = (ubFactor < cols_) ? ubFactor : cols_;
        if (tileElems_ <= 0) {
            tileElems_ = 1;
        }
        tailElems_ = (tileElems_ < cols_) ? (cols_ % tileElems_) : 0;

        const int64_t blockFactor = (tilingData->blockFactor > 0) ? tilingData->blockFactor : 1;
        rowStart_ = static_cast<int64_t>(GetBlockIdx()) * blockFactor;
        rowEnd_ = rowStart_ + blockFactor;
        if (rowEnd_ > rows_) {
            rowEnd_ = rows_;
        }

        // UB blocks are allocated in 32-byte units.
        const int64_t tileBytes = AlignUp32(tileElems_ * static_cast<int64_t>(sizeof(T)));
        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, tileBytes);

        // Byte-valued gather offsets: one table for full tiles and, when the row
        // does not divide evenly, a second table for the trailing tile.
        const int64_t offsetTileBytes = AlignUp32(tileElems_ * static_cast<int64_t>(sizeof(uint32_t)));
        const int64_t offsetTailBytes = (tailElems_ > 0)
            ? AlignUp32(tailElems_ * static_cast<int64_t>(sizeof(uint32_t)))
            : 0;
        pipe_.InitBuffer(offsetBuf_, offsetTileBytes + offsetTailBytes);
        offsetTile_ = offsetBuf_.template Get<uint32_t>();
        if (tailElems_ > 0) {
            offsetTail_ = offsetBuf_.template Get<uint32_t>()[offsetTileBytes /
                                                           static_cast<int64_t>(sizeof(uint32_t))];
        }
        BuildOffsets(offsetTile_, tileElems_);
        if (tailElems_ > 0) {
            BuildOffsets(offsetTail_, tailElems_);
        }
        // The offset tables are written through the scalar pipe and read by
        // vgather; one full barrier in Init is enough because no other pipe is
        // in flight yet.
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        if (cols_ <= 0 || rowStart_ >= rowEnd_) {
            return;
        }

        for (int64_t row = rowStart_; row < rowEnd_; ++row) {
            for (int64_t colStart = 0; colStart < cols_; colStart += tileElems_) {
                const int64_t remain = cols_ - colStart;
                const int64_t count = (remain < tileElems_) ? remain : tileElems_;
                const int64_t srcStart = cols_ - colStart - count;
                if (count == tileElems_) {
                    CopyIn(row, srcStart, count);
                    Compute(offsetTile_, count);
                } else {
                    CopyIn(row, srcStart, count);
                    Compute(offsetTail_, count);
                }
                CopyOut(row, colStart, count);
            }
        }
    }

private:
    static __aicore__ inline int64_t AlignUp32(int64_t value)
    {
        return (value + 31) / 32 * 32;
    }

    __aicore__ inline void BuildOffsets(const LocalTensor<uint32_t>& offsets, int64_t count)
    {
        const uint32_t byteStep = static_cast<uint32_t>(sizeof(T));
        for (int64_t k = 0; k < count; ++k) {
            offsets.SetValue(static_cast<uint32_t>(k),
                             static_cast<uint32_t>(count - 1 - k) * byteStep);
        }
    }

    __aicore__ inline void CopyIn(int64_t row, int64_t srcStart, int64_t count)
    {
        LocalTensor<T> inputLocal = inputQueue_.template AllocTensor<T>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPad(inputLocal, inputGM_[row * cols_ + srcStart], copyParams,
                    {false, 0, 0, static_cast<T>(0)});
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(const LocalTensor<uint32_t>& offsets, int64_t count)
    {
        LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
        LocalTensor<T> outputLocal = outputQueue_.template AllocTensor<T>();
        Gather(outputLocal, inputLocal, offsets, static_cast<uint32_t>(0),
               static_cast<uint32_t>(count));
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(int64_t row, int64_t colStart, int64_t count)
    {
        LocalTensor<T> outputLocal = outputQueue_.template DeQue<T>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPad(outputGM_[row * cols_ + colStart], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    TBuf<QuePosition::VECCALC> offsetBuf_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    LocalTensor<uint32_t> offsetTile_;
    LocalTensor<uint32_t> offsetTail_;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    int64_t tileElems_ = 1;
    int64_t tailElems_ = 0;
    int64_t rowStart_ = 0;
    int64_t rowEnd_ = 0;
};

} // namespace NsFlip

using NsFlip::Flip;

#endif // FLIP_H
