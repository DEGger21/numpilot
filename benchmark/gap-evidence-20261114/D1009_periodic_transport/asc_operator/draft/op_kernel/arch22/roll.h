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
 * \file roll.h
 * \brief Roll (numpy.roll) kernel class definition, arch22 / Ascend910B3.
 *
 * Semantic contract (NumPy oracle, numpy.roll):
 *   out[idx] == in[ (idx with the rolled axis index decreased by shift, modulo axis length) ]
 * Roll is a pure data movement op: every output element is a bit-identical copy of one input
 * element. No arithmetic is performed anywhere, so NaN / +-Inf / +-0.0 / -0.0 and the sign bit
 * are preserved exactly.
 *
 * Index decomposition used by host tiling and by this kernel:
 *   A rank-r tensor rolled along normalized axis a is viewed as three dimensions
 *       outer   = prod(shape[0 : a])          (number of independent cyclic groups)
 *       axisLen = shape[a]                    (length of the rolled axis)
 *       inner   = prod(shape[a+1 : r])        (contiguous run copied together)
 *   With shift normalized into [0, axisLen), each outer group contributes exactly two
 *   contiguous copy segments:
 *     group 0 : out[o, 0            : axisLen-shift, :] <- in[o, shift            : axisLen, :]
 *     group 1 : out[o, axisLen-shift: axisLen      , :] <- in[o, 0                : shift   , :]
 *   so out = concat(in[o, shift:], in[o, :shift]) which is exactly numpy.roll along axis o.
 *   Segments are distributed over cores in a strided fashion (segment s handled by cores
 *   s, s+blockNum, s+2*blockNum, ...), which keeps the short wrap-around segments spread
 *   evenly instead of concentrating them on a few cores.
 *
 * Tiling data fields consumed by this kernel (see op_kernel/arch22/roll_tiling_data.h):
 *   RollTilingData::outer    - number of independent cyclic groups  (prod(shape[0:axis]))
 *   RollTilingData::axisLen  - length of the rolled axis            (shape[axis])
 *   RollTilingData::inner    - contiguous run length copied at once (prod(shape[axis+1:]))
 *   RollTilingData::shift    - already normalized roll amount, 0 <= shift < axisLen
 *   RollTilingData::ubFactor - elements staged per UB tile (0 < ubFactor <= UB budget)
 * Unused-by-kernel bookkeeping fields (totalNum / blockFactor / usedCoreNum) may additionally be
 * present in the tiling struct; they are not required for correctness of this kernel.
 *
 * Supported scope of this candidate (unverified draft, see operator-artifact.json):
 *   float32, 2-D C-contiguous tensors, keyword signature roll(x, shift, axis=<int>).
 *   Anything outside the host-declared support scope is rejected on the host side, never here.
 */

#ifndef ROLL_ARCH22_H
#define ROLL_ARCH22_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "roll_tiling_data.h"
#include "roll_tiling_key.h"

namespace NsRoll {

using namespace AscendC;

// Maximum queue depth (compile-time capacity); the effective depth is BUFFER_NUM below.
constexpr int32_t ROLL_MAX_BUFFER_NUM = 2;

/*!
 * \brief Roll kernel class.
 * \tparam T element type, float32 for the admitted profile.
 * \tparam BUFFER_MODE 0 = single buffer, 1 = double buffer (default).
 *         Both modes are bit-exact; the mode only affects pipelining, never results.
 */
template <typename T = float, int BUFFER_MODE = 1>
class Roll {
    static constexpr int32_t BUFFER_NUM = (BUFFER_MODE == 0) ? 1 : 2;
    // Elements per 32B block; UB->UB DataCopy works on 32B aligned element counts.
    static constexpr int64_t ELEM_ALIGN = static_cast<int64_t>(32 / sizeof(T));

public:
    __aicore__ inline Roll(){};

    /*!
     * \brief Initialize global tensors, tiling parameters and UB queues.
     * \param x GM address of the input tensor (read only).
     * \param y GM address of the output tensor (write only).
     * \param tilingData host tiling data (see the field list in the file header).
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RollTilingData* tilingData);

    /*!
     * \brief Signature-compatible overload; Roll needs no workspace.
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RollTilingData* tilingData);

    /*!
     * \brief CopyIn -> Compute -> CopyOut over every tile of every owned segment.
     */
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t srcOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, ROLL_MAX_BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, ROLL_MAX_BUFFER_NUM> outputQueue;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t outer_ = 0;        // number of independent cyclic groups
    int64_t axisLen_ = 1;      // length of the rolled axis
    int64_t inner_ = 1;        // contiguous run length
    int64_t shift_ = 0;        // normalized roll amount in [0, axisLen_)
    int64_t totalNum_ = 0;     // outer_ * axisLen_ * inner_
    int64_t numSegments_ = 0;  // 2 * outer_ (group 0 and group 1 of every outer group)
    int64_t ubLength_ = 0;     // elements per UB tile, 32B aligned
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Roll<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, const RollTilingData* tilingData)
{
    outer_ = (tilingData->outer > 0) ? tilingData->outer : 0;
    axisLen_ = (tilingData->axisLen > 0) ? tilingData->axisLen : 1;
    inner_ = (tilingData->inner > 0) ? tilingData->inner : 1;
    shift_ = tilingData->shift;

    // Host already normalizes shift; re-normalize defensively so the kernel is self-contained
    // (negative shifts, shift >= axisLen and axisLen == 1 all collapse to a well defined value).
    shift_ = shift_ % axisLen_;
    if (shift_ < 0) {
        shift_ += axisLen_;
    }

    totalNum_ = outer_ * axisLen_ * inner_;
    numSegments_ = 2 * outer_;

    // UB tile size: strictly positive and a multiple of one 32B block so that the aligned
    // UB->UB copy in Compute() can never leave the allocated buffer.
    int64_t ub = tilingData->ubFactor;
    if (ub < ELEM_ALIGN) {
        ub = ELEM_ALIGN;
    }
    ub = ub - (ub % ELEM_ALIGN);
    if (ub < ELEM_ALIGN) {
        ub = ELEM_ALIGN;
    }
    ubLength_ = ub;

    if (totalNum_ > 0) {
        inputGM.SetGlobalBuffer((__gm__ T*)x, totalNum_);
        outputGM.SetGlobalBuffer((__gm__ T*)y, totalNum_);
    }

    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * static_cast<int64_t>(sizeof(T)));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Roll<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                  const RollTilingData* tilingData)
{
    (void)workspace;  // Roll has no workspace requirement; kept for call-site symmetry.
    Init(x, y, tilingData);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Roll<T, BUFFER_MODE>::Process()
{
    if (totalNum_ <= 0) {
        return;  // empty tensor: nothing to copy, output is empty as well
    }

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
    if (blockNum <= 0) {
        return;
    }

    const int64_t headLen = (axisLen_ - shift_) * inner_;  // group 0 length
    const int64_t tailLen = shift_ * inner_;               // group 1 length

    // Strided segment assignment: segment s belongs to core (s % blockNum).
    for (int64_t s = blockIdx; s < numSegments_; s += blockNum) {
        const int64_t outerIdx = s >> 1;   // which cyclic group
        const int64_t group = s & 1;       // 0 = shifted head, 1 = wrapped tail
        const int64_t len = (group == 0) ? headLen : tailLen;
        if (len <= 0) {
            continue;  // zero length group (e.g. shift == 0 or shift == axisLen after normalization)
        }

        const int64_t groupBase = outerIdx * axisLen_ * inner_;
        const int64_t shiftBase = shift_ * inner_;
        int64_t dstBase = 0;
        int64_t srcBase = 0;
        if (group == 0) {
            dstBase = groupBase;
            srcBase = groupBase + shiftBase;
        } else {
            dstBase = groupBase + headLen;
            srcBase = groupBase;
        }

        for (int64_t off = 0; off < len; off += ubLength_) {
            int64_t currentNum = len - off;
            if (currentNum > ubLength_) {
                currentNum = ubLength_;
            }
            CopyIn(srcBase + off, currentNum);
            Compute(currentNum);
            CopyOut(dstBase + off, currentNum);
        }
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Roll<T, BUFFER_MODE>::CopyIn(int64_t srcOffset, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueue.template AllocTensor<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    // Always C-contiguous on the admitted profile: one contiguous run, no stride handling,
    // no padding. DataCopyPad accepts byte lengths that are not 32B multiples.
    AscendC::DataCopyPad(xLocal, inputGM[srcOffset], copyParams, {false, 0, 0, static_cast<T>(0)});
    inputQueue.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Roll<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueue.template DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueue.template AllocTensor<T>();

    // Bit-exact copy. No arithmetic is used on purpose: any Add/Mul with 0 would turn -0.0
    // into +0.0 and would violate the NumPy signed-zero contract. UB->UB DataCopy moves raw
    // 32B blocks, so NaN payloads, +-Inf and the sign bit survive unchanged.
    // The count is rounded up to a full 32B block; the trailing lanes only ever live inside the
    // UB tile and are never written back to GM (CopyOut uses the exact byte length).
    int64_t alignedNum = currentNum;
    const int64_t remainder = alignedNum % ELEM_ALIGN;
    if (remainder != 0) {
        alignedNum += (ELEM_ALIGN - remainder);
    }
    if (alignedNum > ubLength_) {
        alignedNum = ubLength_;  // cannot happen: currentNum <= ubLength_ and ubLength_ is aligned
    }

    AscendC::DataCopy(zLocal, xLocal, static_cast<uint32_t>(alignedNum));

    outputQueue.template EnQue<T>(zLocal);
    inputQueue.FreeTensor(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Roll<T, BUFFER_MODE>::CopyOut(int64_t dstOffset, int64_t currentNum)
{
    AscendC::LocalTensor<T> zLocal = outputQueue.template DeQue<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    AscendC::DataCopyPad(outputGM[dstOffset], zLocal, copyParams);
    outputQueue.FreeTensor(zLocal);
}

}  // namespace NsRoll

#endif  // ROLL_ARCH22_H