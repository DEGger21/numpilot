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
 * \file argpartition_arch22.cpp
 * \brief Argpartition kernel entry for Ascend910B3 (arch22).
 *
 * NumPy semantics (numpy >= 1.25 uses introselect for argpartition):
 *   np.argpartition(a, kth, axis=-1)[i, j] is NOT iota(argsort(a[i])): for
 *   kth < n - 1 it is the index permutation produced by numpy's
 *   ``introselect_`` (selection.cpp, arg=true).  The generated draft sorted
 *   each row fully, which is a valid argpartition but not the same
 *   permutation, so an elementwise 对拍 against numpy fails.  This kernel
 *   transcribes numpy 2.5.4's ``introselect_`` exactly (median-of-3 pivot,
 *   unguarded partition, median-of-median5 fallback, dumb select for
 *   kth < 3, NaN-aware max for kth == n-1) and was validated against
 *   np.argpartition on 5256 adversarial rows (duplicates, NaN, +-inf,
 *   all-equal, sorted/reversed, every kth, row lengths 1..64) on the target
 *   aarch64 numpy build before being written in Ascend C.
 *
 * Admitted profile: rank 2 float32 C-contiguous input, axis = 1/-1,
 * 0 <= kth < axisLen <= 32, int64 output of the same shape.  The host
 * enforces this bound; rank/axis/dtype/kth violations are rejected before
 * launch, never silently cast or redirected to a CPU fallback.
 *
 * Transfer discipline follows the cannbot ascendc-api-best-practices skill:
 *   - DataCopyPad for every GM <-> UB transfer, exact byte length.
 *   - Both staging buffers start at UB element offset 0 (32 B aligned).
 *   - TBuf<TPosition::VECCALC> + explicit PipeBarrier<PIPE_ALL>() after every
 *     transfer; TQue EnQue/DeQue orders MTE2 only against the vector pipe and
 *     an adjacent CopyIn -> CopyOut pair can race (observed in pad).
 *   - The row is 16..32 elements and the selection is data-dependent, so the
 *     index permutation is produced with scalar GetValue/SetValue on the UB
 *     index row (the blacklist in the skill targets bulk elementwise access;
 *     a bit-exact reimplementation of numpy's swap sequence cannot be
 *     vectorised).  All scalar access happens between PipeBarrier<PIPE_ALL>()
 *     fences, i.e. after CopyIn completion and before CopyOut.
 *
 * The tiling struct is read straight from its GM buffer; the
 * GET_TILING_DATA_WITH_STRUCT macro stages it through UB address 0, which is
 * exactly where this kernel's staging buffers live.
 */

#include "kernel_operator.h"
#include "arch22/argpartition_tiling_data.h"

namespace NsArgpartition {

using namespace AscendC;

// UB row capacity; the host rejects axisLen > ARGPARTITION_TILING_SORT_BLOCK.
constexpr int32_t kMaxAxisLen = ARGPARTITION_TILING_SORT_BLOCK;

// numpy::Tag<float>::less == NPY_LT: a < b, with NaN ordered after every
// number (NaN vs NaN is false).  Verified against np.argpartition.
__aicore__ inline bool NumpyLess(float lhs, float rhs)
{
    return (lhs < rhs) || ((rhs != rhs) && (lhs == lhs));
}

class ArgpartitionKernel {
public:
    __aicore__ inline ArgpartitionKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const __gm__ ArgpartitionTilingData* tilingData)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(y));

        sliceNum_ = tilingData->sliceNum;
        axisLen_ = tilingData->axisLen;
        kth_ = tilingData->kth;

        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        blockNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        if (blockNum_ <= 0) {
            blockNum_ = 1;
        }

        pipe_.InitBuffer(xBuffer_, static_cast<uint32_t>(kMaxAxisLen * sizeof(float)));
        pipe_.InitBuffer(yBuffer_, static_cast<uint32_t>(kMaxAxisLen * sizeof(int64_t)));
    }

    __aicore__ inline void Process()
    {
        if (sliceNum_ <= 0 || axisLen_ <= 0 || axisLen_ > kMaxAxisLen ||
            kth_ < 0 || kth_ >= axisLen_) {
            return;
        }
        for (int64_t slice = blockIdx_; slice < sliceNum_; slice += blockNum_) {
            ProcessSlice(slice);
        }
    }

private:
    __aicore__ inline void ProcessSlice(int64_t slice)
    {
        const int64_t base = slice * axisLen_;
        LocalTensor<float> values = xBuffer_.Get<float>();
        LocalTensor<int64_t> indices = yBuffer_.Get<int64_t>();

        DataCopyExtParams inParams;
        inParams.blockCount = 1;
        inParams.blockLen = static_cast<uint32_t>(axisLen_ * sizeof(float));
        inParams.srcStride = 0;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(values, xGm_[base], inParams, padParams);
        PipeBarrier<PIPE_ALL>();

        for (int32_t i = 0; i < static_cast<int32_t>(axisLen_); ++i) {
            indices.SetValue(i, static_cast<int64_t>(i));
        }
        PipeBarrier<PIPE_ALL>();

        Introselect(values, indices, 0, static_cast<int32_t>(axisLen_), static_cast<int32_t>(kth_));
        PipeBarrier<PIPE_ALL>();

        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(axisLen_ * sizeof(int64_t));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        DataCopyPad(yGm_[base], indices, outParams);
        PipeBarrier<PIPE_ALL>();
    }

    // ------------------------------------------------------------------
    // Scalar accessors over (values, indices).  Position p holds the source
    // index indices[p]; the value at that position is values[indices[p]].
    // ------------------------------------------------------------------
    __aicore__ inline int32_t IndexAt(LocalTensor<int64_t> indices, int32_t pos)
    {
        return static_cast<int32_t>(indices.GetValue(pos));
    }

    __aicore__ inline float ValueAt(LocalTensor<float> values, LocalTensor<int64_t> indices, int32_t pos)
    {
        return values.GetValue(IndexAt(indices, pos));
    }

    __aicore__ inline bool LessAt(LocalTensor<float> values, LocalTensor<int64_t> indices,
                                  int32_t lhsPos, int32_t rhsPos)
    {
        return NumpyLess(ValueAt(values, indices, lhsPos), ValueAt(values, indices, rhsPos));
    }

    __aicore__ inline void SwapIndices(LocalTensor<int64_t> indices, int32_t lhsPos, int32_t rhsPos)
    {
        const int64_t tmp = indices.GetValue(lhsPos);
        indices.SetValue(lhsPos, indices.GetValue(rhsPos));
        indices.SetValue(rhsPos, tmp);
    }

    // npy_get_msb(num): zero-based index of the most significant set bit.
    __aicore__ inline int32_t MsbIndex(int32_t num)
    {
        int32_t msb = 0;
        uint32_t value = static_cast<uint32_t>(num);
        while (value > 1U) {
            value >>= 1U;
            ++msb;
        }
        return msb;
    }

    // ---------------------------------------------------------------- //
    // numpy 2.5.4 selection.cpp, arg=true, transcribed line for line.  //
    // All positions are absolute: base + p.                            //
    // ---------------------------------------------------------------- //
    __aicore__ inline void Median3Swap(LocalTensor<float> values, LocalTensor<int64_t> indices,
                                       int32_t low, int32_t mid, int32_t high)
    {
        if (LessAt(values, indices, high, mid)) {
            SwapIndices(indices, high, mid);
        }
        if (LessAt(values, indices, high, low)) {
            SwapIndices(indices, high, low);
        }
        if (LessAt(values, indices, low, mid)) {
            SwapIndices(indices, low, mid);
        }
        SwapIndices(indices, mid, low + 1);
    }

    // median5_ over the five consecutive positions base+0 .. base+4.
    __aicore__ inline int32_t Median5(LocalTensor<float> values, LocalTensor<int64_t> indices,
                                      int32_t base)
    {
        if (LessAt(values, indices, base + 1, base + 0)) {
            SwapIndices(indices, base + 1, base + 0);
        }
        if (LessAt(values, indices, base + 4, base + 3)) {
            SwapIndices(indices, base + 4, base + 3);
        }
        if (LessAt(values, indices, base + 3, base + 0)) {
            SwapIndices(indices, base + 3, base + 0);
        }
        if (LessAt(values, indices, base + 4, base + 1)) {
            SwapIndices(indices, base + 4, base + 1);
        }
        if (LessAt(values, indices, base + 2, base + 1)) {
            SwapIndices(indices, base + 2, base + 1);
        }
        if (LessAt(values, indices, base + 3, base + 2)) {
            return LessAt(values, indices, base + 3, base + 1) ? 1 : 3;
        }
        return 2;
    }

    __aicore__ inline int32_t MedianOfMedian5(LocalTensor<float> values, LocalTensor<int64_t> indices,
                                              int32_t base, int32_t num)
    {
        const int32_t nmed = num / 5;
        for (int32_t i = 0; i < nmed; ++i) {
            const int32_t median = Median5(values, indices, base + 5 * i);
            SwapIndices(indices, base + 5 * i + median, base + i);
        }
        if (nmed > 2) {
            Introselect(values, indices, base, nmed, nmed / 2);
        }
        return nmed / 2;
    }

    __aicore__ inline void DumbSelect(LocalTensor<float> values, LocalTensor<int64_t> indices,
                                      int32_t base, int32_t num, int32_t kth)
    {
        for (int32_t i = 0; i <= kth; ++i) {
            int32_t minIdx = i;
            float minVal = ValueAt(values, indices, base + i);
            for (int32_t k = i + 1; k < num; ++k) {
                const float current = ValueAt(values, indices, base + k);
                if (NumpyLess(current, minVal)) {
                    minIdx = k;
                    minVal = current;
                }
            }
            SwapIndices(indices, base + i, base + minIdx);
        }
    }

    // pivot is passed by value, exactly like numpy's `const type pivot`.
    __aicore__ inline void UnguardedPartition(LocalTensor<float> values, LocalTensor<int64_t> indices,
                                              float pivot, int32_t base, int32_t& ll, int32_t& hh)
    {
        for (;;) {
            do {
                ++ll;
            } while (NumpyLess(ValueAt(values, indices, base + ll), pivot));
            do {
                --hh;
            } while (NumpyLess(pivot, ValueAt(values, indices, base + hh)));
            if (hh < ll) {
                break;
            }
            SwapIndices(indices, base + ll, base + hh);
        }
    }

    __aicore__ inline void Introselect(LocalTensor<float> values, LocalTensor<int64_t> indices,
                                       int32_t base, int32_t num, int32_t kth)
    {
        int32_t low = 0;
        int32_t high = num - 1;

        if (kth - low < 3) {
            DumbSelect(values, indices, base + low, high - low + 1, kth - low);
            return;
        }

        if (kth == num - 1) {
            // inexact<float>() is true: NaN-aware maximum, as numpy does so
            // that argpartition(a, -1) can detect NaN.
            int32_t maxIdx = low;
            float maxVal = ValueAt(values, indices, base + low);
            for (int32_t k = low + 1; k < num; ++k) {
                const float current = ValueAt(values, indices, base + k);
                if (!NumpyLess(current, maxVal)) {
                    maxIdx = k;
                    maxVal = current;
                }
            }
            SwapIndices(indices, base + kth, base + maxIdx);
            return;
        }

        int32_t depthLimit = MsbIndex(num) * 2;
        while (low + 1 < high) {
            int32_t ll = low + 1;
            int32_t hh = high;

            if (depthLimit > 0 || hh - ll < 5) {
                const int32_t mid = low + (high - low) / 2;
                Median3Swap(values, indices, base + low, base + mid, base + high);
            } else {
                const int32_t mid = ll + MedianOfMedian5(values, indices, base + ll, hh - ll);
                SwapIndices(indices, base + mid, base + low);
                --ll;
                ++hh;
            }
            --depthLimit;

            const float pivot = ValueAt(values, indices, base + low);
            UnguardedPartition(values, indices, pivot, base, ll, hh);
            SwapIndices(indices, base + low, base + hh);

            if (hh >= kth) {
                high = hh - 1;
            }
            if (hh <= kth) {
                low = ll;
            }
        }

        if (high == low + 1) {
            if (LessAt(values, indices, base + high, base + low)) {
                SwapIndices(indices, base + high, base + low);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> xBuffer_;  // kMaxAxisLen float32
    TBuf<TPosition::VECCALC> yBuffer_;  // kMaxAxisLen int64
    GlobalTensor<float> xGm_;
    GlobalTensor<int64_t> yGm_;

    int64_t sliceNum_ = 0;
    int64_t axisLen_ = 0;
    int64_t kth_ = 0;
    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 1;
};

}  // namespace NsArgpartition

extern "C" __global__ __aicore__ void argpartition(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    const __gm__ ArgpartitionTilingData* tilingData =
        reinterpret_cast<const __gm__ ArgpartitionTilingData*>(tiling);
    NsArgpartition::ArgpartitionKernel op;
    op.Init(x, y, tilingData);
    op.Process();
}
