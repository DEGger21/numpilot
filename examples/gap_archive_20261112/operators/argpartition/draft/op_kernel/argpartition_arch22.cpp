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
 * The admitted profile is float32, C-contiguous input with the partition axis
 * being the last dimension.  The observed numpy.argpartition profile is
 * [32, 16] with axis=1 and kth=3.  For axis length <= 16 NumPy's selection
 * fallback performs a stable insertion sort; this kernel uses the same stable
 * (value, index) ordering, which preserves the oracle's deterministic order
 * for the observed length of 16 while remaining a valid argpartition for
 * longer rows.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

struct ArgpartitionTilingData {
    int64_t totalSlices = 0;  // Number of independent partition rows.
    int64_t axisLength = 0;   // Number of elements in each partition row.
    int64_t kth = 0;          // Normalized-or-original kth; host validates it.
    int64_t axis = 0;         // Normalized axis; host rejects non-last-axis rows.
};

namespace NsArgpartition {

using namespace AscendC;

__aicore__ inline bool ArgpartitionValueLess(float lhs, float rhs)
{
    const bool lhsNan = lhs != lhs;
    const bool rhsNan = rhs != rhs;
    if (lhsNan) {
        return false;
    }
    if (rhsNan) {
        return true;
    }
    return lhs < rhs;
}

__aicore__ inline int64_t AlignUp32(int64_t value)
{
    return (value + 31) / 32 * 32;
}

class ArgpartitionKernel {
public:
    __aicore__ inline ArgpartitionKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ArgpartitionTilingData* tilingData)
    {
        totalSlices_ = tilingData->totalSlices;
        axisLength_ = tilingData->axisLength;
        kth_ = tilingData->kth;
        axis_ = tilingData->axis;
        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        blockNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        if (blockNum_ <= 0) {
            blockNum_ = 1;
        }

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(y));

        if (axisLength_ > 0) {
            const int64_t inputBytes =
                AlignUp32(axisLength_ * static_cast<int64_t>(sizeof(float)));
            const int64_t outputBytes =
                AlignUp32(axisLength_ * static_cast<int64_t>(sizeof(int64_t)));
            pipe_.InitBuffer(inputQueue_, 1, static_cast<int32_t>(inputBytes));
            pipe_.InitBuffer(outputQueue_, 1, static_cast<int32_t>(outputBytes));
        }
    }

    __aicore__ inline void Process()
    {
        if (totalSlices_ <= 0 || axisLength_ <= 0) {
            return;
        }

        int64_t normalizedKth = kth_;
        if (normalizedKth < 0) {
            normalizedKth += axisLength_;
        }
        if (normalizedKth < 0 || normalizedKth >= axisLength_) {
            return;
        }
        (void)axis_;

        for (int64_t slice = blockIdx_; slice < totalSlices_; slice += blockNum_) {
            ProcessSlice(slice);
        }
    }

private:
    __aicore__ inline void ProcessSlice(int64_t slice)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = static_cast<uint32_t>(axisLength_ * sizeof(float));
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;

        LocalTensor<float> xLocal = inputQueue_.AllocTensor<float>();
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(xLocal, xGm_[slice * axisLength_], copyInParams, padParams);
        inputQueue_.EnQue(xLocal);
        xLocal = inputQueue_.DeQue<float>();
        AscendC::PipeBarrier<PIPE_ALL>();

        LocalTensor<int64_t> yLocal = outputQueue_.AllocTensor<int64_t>();
        for (int64_t i = 0; i < axisLength_; ++i) {
            yLocal.SetValue(static_cast<int32_t>(i), i);
        }

        // Stable insertion sort by value.  This matches the selection fallback
        // used by numpy.argpartition for the observed axis length of 16.
        for (int64_t i = 1; i < axisLength_; ++i) {
            const float key = xLocal.GetValue(static_cast<int32_t>(i));
            int64_t j = i - 1;
            while (j >= 0) {
                const int64_t previousIndex =
                    yLocal.GetValue(static_cast<int32_t>(j));
                const float previousValue =
                    xLocal.GetValue(static_cast<int32_t>(previousIndex));
                if (!ArgpartitionValueLess(key, previousValue)) {
                    break;
                }
                yLocal.SetValue(static_cast<int32_t>(j + 1), previousIndex);
                --j;
            }
            yLocal.SetValue(static_cast<int32_t>(j + 1), i);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        outputQueue_.EnQue(yLocal);
        yLocal = outputQueue_.DeQue<int64_t>();

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen =
            static_cast<uint32_t>(axisLength_ * sizeof(int64_t));
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        copyOutParams.rsv = 0;
        DataCopyPad(yGm_[slice * axisLength_], yLocal, copyOutParams);

        outputQueue_.FreeTensor(yLocal);
        inputQueue_.FreeTensor(xLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;

    GlobalTensor<float> xGm_;
    GlobalTensor<int64_t> yGm_;

    int64_t totalSlices_ = 0;
    int64_t axisLength_ = 0;
    int64_t kth_ = 0;
    int64_t axis_ = 0;
    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 1;
};

}  // namespace NsArgpartition

extern "C" __global__ __aicore__ void argpartition(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(ArgpartitionTilingData);
    GET_TILING_DATA_WITH_STRUCT(ArgpartitionTilingData, tilingData, tiling);

    NsArgpartition::ArgpartitionKernel op;
    op.Init(x, y, &tilingData);
    op.Process();
}