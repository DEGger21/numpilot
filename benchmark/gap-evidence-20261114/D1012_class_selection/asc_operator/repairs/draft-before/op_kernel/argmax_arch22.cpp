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
 * \file argmax_arch22.cpp
 * \brief Argmax kernel entry for arch22.
 *
 * Bounded support for the observed profile only:
 *   input  : float32, shape [1024, 1024], C-contiguous
 *   axis   : 1
 *   output : int64, shape [1024]
 *
 * NumPy argmax semantics implemented here:
 *   - return the first occurrence of the maximum value
 *   - treat NaN as greater than every non-NaN value
 *   - keep the first NaN when multiple NaNs are present
 *   - +0.0 and -0.0 compare equal, so the first of the two is returned
 *
 * The host side is responsible for rejecting any dtype, shape, rank, layout,
 * or axis outside this bounded profile. This kernel intentionally does not
 * silently reinterpret unsupported inputs.
 */

#ifndef ARGMAX_ARCH22_CPP
#define ARGMAX_ARCH22_CPP

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch22/argmax_tiling_data.h"

#include <cstdint>
#include <type_traits>

namespace NsArgmaxArch22 {

using namespace AscendC;

constexpr int64_t kArgmaxRows = 1024;
constexpr int64_t kArgmaxCols = 1024;
constexpr int64_t kOutputBufferBytes = 32;

__aicore__ inline bool ArgmaxIsNanScalar(float value)
{
    union FloatBits {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    const uint32_t exponent = bits.u & 0x7F800000U;
    const uint32_t mantissa = bits.u & 0x007FFFFFU;
    return (exponent == 0x7F800000U) && (mantissa != 0U);
}

template <typename D_T_X>
class ArgmaxArch22Kernel {
    static_assert(std::is_same<D_T_X, float>::value,
                  "argmax arch22 kernel supports float32 input only");

public:
    __aicore__ inline ArgmaxArch22Kernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ D_T_X*>(x), kArgmaxRows * kArgmaxCols);
        outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(y), kArgmaxRows);

        pipe_.InitBuffer(inputQueue_, 1, kArgmaxCols * static_cast<int64_t>(sizeof(D_T_X)));
        pipe_.InitBuffer(outputQueue_, 1, kOutputBufferBytes);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        if (blockNum <= 0) {
            return;
        }

        for (int64_t row = blockIdx; row < kArgmaxRows; row += blockNum) {
            CopyIn(row);
            Compute();
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t row)
    {
        LocalTensor<D_T_X> xLocal = inputQueue_.AllocTensor<D_T_X>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(kArgmaxCols * static_cast<int64_t>(sizeof(D_T_X)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        const uint32_t inputOffset = static_cast<uint32_t>(row * kArgmaxCols);
        DataCopyPad(xLocal, inputGM_[inputOffset], copyParams,
                    {false, 0, 0, static_cast<D_T_X>(0)});
        inputQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<D_T_X> xLocal = inputQueue_.DeQue<D_T_X>();

        float bestValue = static_cast<float>(xLocal.GetValue(0));
        int64_t bestIndex = 0;
        bool bestIsNan = ArgmaxIsNanScalar(bestValue);

        for (int64_t col = 1; col < kArgmaxCols; ++col) {
            const float currentValue = static_cast<float>(xLocal.GetValue(static_cast<uint32_t>(col)));
            const bool currentIsNan = ArgmaxIsNanScalar(currentValue);

            bool shouldReplace = false;
            if (currentIsNan) {
                shouldReplace = !bestIsNan;
            } else if (!bestIsNan) {
                shouldReplace = currentValue > bestValue;
            }

            if (shouldReplace) {
                bestValue = currentValue;
                bestIndex = col;
                bestIsNan = currentIsNan;
            }
        }

        LocalTensor<int64_t> outLocal = outputQueue_.AllocTensor<int64_t>();
        outLocal.SetValue(0, bestIndex);
        outputQueue_.EnQue(outLocal);

        inputQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t row)
    {
        LocalTensor<int64_t> outLocal = outputQueue_.DeQue<int64_t>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(sizeof(int64_t));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        const uint32_t outputOffset = static_cast<uint32_t>(row);
        DataCopyPad(outputGM_[outputOffset], outLocal, copyParams);
        outputQueue_.FreeTensor(outLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;

    GlobalTensor<D_T_X> inputGM_;
    GlobalTensor<int64_t> outputGM_;
};

} // namespace NsArgmaxArch22

template <typename D_T_X>
__global__ __aicore__ void argmax(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ArgmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(ArgmaxTilingData, tilingData, tiling);
    (void)tilingData;
    (void)workspace;

    NsArgmaxArch22::ArgmaxArch22Kernel<D_T_X> op;
    op.Init(x, y);
    op.Process();
}

#endif // ARGMAX_ARCH22_CPP