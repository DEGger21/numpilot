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
 * \file repeat.h
 * \brief Repeat (numpy.repeat) kernel class definition, arch22 (Ascend910B3).
 *
 * Semantics implemented here (accuracy first, no fast path):
 *   Out[r, c] = In[r, c / repeats] for c in [0, inner * repeats)
 * where
 *   rows      : product of all output dimensions except the repeated axis
 *   inner     : length of the input along the repeated axis
 *   repeats   : number of copies of every element along the repeated axis
 *   outInner  : inner * repeats (length of the output along the repeated axis)
 * An elementwise copy keeps every float32 bit pattern (NaN / +-Inf / +-0.0) intact.
 *
 * TilingData contract (single authoritative definition lives in
 * op_kernel/arch22/repeat_tiling_data.h, shared with op_host/arch22/repeat_tiling.cpp):
 *   struct RepeatTilingData {
 *       int64_t rows;         // output rows (product of non-repeated dims)
 *       int64_t inner;        // input length along the repeated axis
 *       int64_t repeats;      // copies per element, >= 1
 *       int64_t blockFactor;  // rows handled by one AI core
 *       int64_t ubFactor;     // max output elements per UB tile (host guarantees
 *                             // ubFactor >= repeats and the UB budget is respected)
 *   };
 *
 * Template parameters match ASCENDC_TPL_ARGS_DECL in repeat_tiling_key.h exactly:
 *   - T           : element type, float32 only for the admitted scope
 *   - BUFFER_MODE : 0 = single buffer, 1 = double buffer scheduling knob
 * No TQue is used (only TBuf plus explicit pipe synchronization), therefore TPipe
 * never allocates an event id and the fixed sync event id below cannot collide.
 */

#ifndef REPEAT_ARCH22_H
#define REPEAT_ARCH22_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "repeat_tiling_data.h"
#include "repeat_tiling_key.h"

namespace NsRepeat {

using namespace AscendC;

// UB allocation granularity (bytes).
constexpr int64_t REPEAT_ALIGN_BYTES = 32;
// Fixed synchronization event id: safe because this kernel does not create any TQue.
constexpr int32_t REPEAT_SYNC_EVENT_ID = 0;

__aicore__ inline int64_t RepeatAlignUp(int64_t value, int64_t align)
{
    if (align <= 0) {
        return value;
    }
    return (value + align - 1) / align * align;
}

template <typename T, int BUFFER_MODE>
class Repeat {
public:
    __aicore__ inline Repeat() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RepeatTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t srcOffset, int64_t count);
    __aicore__ inline void Compute(int64_t countIn);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t count);

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> outputBuf_;
    LocalTensor<T> inputLocal_;
    LocalTensor<T> outputLocal_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    int64_t rowStart_ = 0;   // first output row owned by this core
    int64_t rowCount_ = 0;   // number of output rows owned by this core
    int64_t inner_ = 0;      // input length along the repeated axis
    int64_t repeats_ = 1;    // copies per element
    int64_t outInner_ = 0;   // inner_ * repeats_
    int64_t chunkIn_ = 0;    // input elements per UB tile
    int64_t chunkOut_ = 0;   // output elements per UB tile (chunkIn_ * repeats_)
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Repeat<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, const RepeatTilingData* tilingData)
{
    static_assert((BUFFER_MODE == 0) || (BUFFER_MODE == 1),
                  "BUFFER_MODE must be 0 (single buffer) or 1 (double buffer) as declared by ASCENDC_TPL_ARGS_DECL");

    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t blockFactor = (tilingData->blockFactor > 0) ? tilingData->blockFactor : 0;

    rowStart_ = blockIdx * blockFactor;
    int64_t remainRows = tilingData->outerSize - rowStart_;
    if (remainRows < 0) {
        remainRows = 0;
    }
    rowCount_ = (remainRows > blockFactor) ? blockFactor : remainRows;

    inner_ = (tilingData->axisSize > 0) ? tilingData->axisSize : 0;
    repeats_ = (tilingData->repeats > 0) ? tilingData->repeats : 1;
    outInner_ = inner_ * repeats_;

    // UB tile: ubFactor counts output elements, so one tile consumes ubFactor / repeats input elements.
    int64_t maxChunkIn = tilingData->ubFactor / repeats_;
    if (maxChunkIn < 1) {
        maxChunkIn = 1;
    }
    if ((inner_ > 0) && (maxChunkIn > inner_)) {
        maxChunkIn = inner_;
    }
    if (maxChunkIn < 1) {
        maxChunkIn = 1;
    }
    chunkIn_ = maxChunkIn;
    chunkOut_ = chunkIn_ * repeats_;

    const int64_t inputBytes = RepeatAlignUp(chunkIn_ * static_cast<int64_t>(sizeof(T)), REPEAT_ALIGN_BYTES);
    const int64_t outputBytes = RepeatAlignUp(chunkOut_ * static_cast<int64_t>(sizeof(T)), REPEAT_ALIGN_BYTES);
    pipe_.InitBuffer(inputBuf_, static_cast<uint32_t>(inputBytes));
    pipe_.InitBuffer(outputBuf_, static_cast<uint32_t>(outputBytes));

    inputLocal_ = inputBuf_.Get<T>();
    outputLocal_ = outputBuf_.Get<T>();
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Repeat<T, BUFFER_MODE>::Process()
{
    for (int64_t i = 0; i < rowCount_; ++i) {
        const int64_t row = rowStart_ + i;
        const int64_t srcRowOffset = row * inner_;
        const int64_t dstRowOffset = row * outInner_;
        for (int64_t colIn = 0; colIn < inner_; colIn += chunkIn_) {
            int64_t currentIn = inner_ - colIn;
            if (currentIn > chunkIn_) {
                currentIn = chunkIn_;
            }
            const int64_t currentOut = currentIn * repeats_;
            CopyIn(srcRowOffset + colIn, currentIn);
            Compute(currentIn);
            CopyOut(dstRowOffset + colIn * repeats_, currentOut);
        }
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Repeat<T, BUFFER_MODE>::CopyIn(int64_t srcOffset, int64_t count)
{
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(inputLocal_, inputGm_[srcOffset], copyParams, padParams);

    // MTE2 -> S: the scalar loop below reads UB directly, so the copy must be visible first.
    event_t syncEvent = static_cast<event_t>(REPEAT_SYNC_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(syncEvent);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(syncEvent);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Repeat<T, BUFFER_MODE>::Compute(int64_t countIn)
{
    // Accuracy-first exact replication: every source element is copied `repeats_` times.
    // No arithmetic is applied, so float32 special values and signed zero are preserved bit-exactly.
    for (int64_t i = 0; i < countIn; ++i) {
        const T value = inputLocal_.GetValue(static_cast<uint32_t>(i));
        const int64_t dstBase = i * repeats_;
        for (int64_t r = 0; r < repeats_; ++r) {
            outputLocal_.SetValue(static_cast<uint32_t>(dstBase + r), value);
        }
    }

    // S -> MTE3: scalar results must be visible before the UB -> GM copy starts.
    event_t syncEvent = static_cast<event_t>(REPEAT_SYNC_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(syncEvent);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(syncEvent);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Repeat<T, BUFFER_MODE>::CopyOut(int64_t dstOffset, int64_t count)
{
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(outputGm_[dstOffset], outputLocal_, copyParams);

    // MTE3 -> S: the next tile writes the output UB buffer again, so the copy must have finished reading it.
    event_t syncEvent = static_cast<event_t>(REPEAT_SYNC_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(syncEvent);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(syncEvent);
}

} // namespace NsRepeat

#endif // REPEAT_ARCH22_H