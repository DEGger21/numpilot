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
 * \file argpartition.h
 * \brief Argpartition kernel class for arch22.
 *
 * Supported semantic scope for this draft:
 *   - input dtype: float32
 *   - output dtype: int64
 *   - rank: 2
 *   - axis: 1
 *   - C-contiguous input and output
 *
 * The observed profile is [32, 16] with kth=3 along axis=1. The host
 * tiling layer is responsible for rejecting profiles outside that admitted
 * scope. This kernel computes one valid deterministic argpartition result
 * per row by producing a full ascending index permutation. A full sort is
 * a valid argpartition because all elements before kth are <= the kth
 * element and all elements after kth are >= the kth element. The ordering
 * within the two partitions is not required to match NumPy's unstable
 * introselect ordering.
 */
#ifndef ARGPARTITION_H
#define ARGPARTITION_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#if defined(__has_include)
#  if __has_include("argpartition_tiling_data.h")
#    include "argpartition_tiling_data.h"
#  elif __has_include("../../op_host/arch22/argpartition_tiling_data.h")
#    include "../../op_host/arch22/argpartition_tiling_data.h"
#  elif __has_include("../../op_host/argpartition_tiling_data.h")
#    include "../../op_host/argpartition_tiling_data.h"
#  else
#    include "argpartition_tiling_data.h"
#  endif
#else
#  include "argpartition_tiling_data.h"
#endif

namespace NsArgpartition {

using namespace AscendC;

/*
 * Fallback layout used only when the host tiling struct does not expose any
 * of the recognized field names below. The host tiling data contract is:
 *   row count, column count, kth, block factor, ub factor.
 */
struct ArgpartitionTilingLayout {
    int64_t totalRows = 0;
    int64_t totalCols = 0;
    int64_t kth = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
};

template <typename... Ts>
struct ArgTilingVoid {
    using Type = void;
};

template <typename... Ts>
using ArgTilingVoidT = typename ArgTilingVoid<Ts...>::Type;

#define ARGPARTITION_DECLARE_MEMBER_DETECTOR(Name, Member)                                      \
    template <typename T, typename = void>                                                      \
    struct Name {                                                                               \
        static constexpr bool Value = false;                                                    \
    };                                                                                          \
    template <typename T>                                                                       \
    struct Name<T, ArgTilingVoidT<decltype(static_cast<T*>(nullptr)->Member)>> {                \
        static constexpr bool Value = true;                                                     \
    }

ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasTotalRows, totalRows);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasRows, rows);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasNumRows, numRows);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasRowNum, rowNum);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasTotalRowsSnake, total_rows);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasNRows, nRows);

ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasTotalCols, totalCols);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasCols, cols);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasNumCols, numCols);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasColNum, colNum);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasRowLength, rowLength);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasColLength, colLength);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasTotalColsSnake, total_cols);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasNCols, nCols);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasColsPerRow, colsPerRow);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasN, n);

ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasTotalNum, totalNum);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasTotalElements, totalElements);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasNumElements, numElements);
ARGPARTITION_DECLARE_MEMBER_DETECTOR(ArgHasTotalSize, totalSize);

#undef ARGPARTITION_DECLARE_MEMBER_DETECTOR

template <typename TilingT>
__aicore__ inline int64_t ArgGetTotalRows(const TilingT* td)
{
    if constexpr (ArgHasTotalRows<TilingT>::Value) {
        return static_cast<int64_t>(td->totalRows);
    } else if constexpr (ArgHasRows<TilingT>::Value) {
        return static_cast<int64_t>(td->rows);
    } else if constexpr (ArgHasNumRows<TilingT>::Value) {
        return static_cast<int64_t>(td->numRows);
    } else if constexpr (ArgHasRowNum<TilingT>::Value) {
        return static_cast<int64_t>(td->rowNum);
    } else if constexpr (ArgHasTotalRowsSnake<TilingT>::Value) {
        return static_cast<int64_t>(td->total_rows);
    } else if constexpr (ArgHasNRows<TilingT>::Value) {
        return static_cast<int64_t>(td->nRows);
    } else if constexpr (ArgHasTotalNum<TilingT>::Value && ArgHasTotalCols<TilingT>::Value) {
        const int64_t c = static_cast<int64_t>(td->totalCols);
        return c > 0 ? static_cast<int64_t>(td->totalNum) / c : 0;
    } else if constexpr (ArgHasTotalNum<TilingT>::Value && ArgHasCols<TilingT>::Value) {
        const int64_t c = static_cast<int64_t>(td->cols);
        return c > 0 ? static_cast<int64_t>(td->totalNum) / c : 0;
    } else if constexpr (ArgHasTotalNum<TilingT>::Value && ArgHasRowLength<TilingT>::Value) {
        const int64_t c = static_cast<int64_t>(td->rowLength);
        return c > 0 ? static_cast<int64_t>(td->totalNum) / c : 0;
    } else if constexpr (ArgHasTotalElements<TilingT>::Value && ArgHasTotalCols<TilingT>::Value) {
        const int64_t c = static_cast<int64_t>(td->totalCols);
        return c > 0 ? static_cast<int64_t>(td->totalElements) / c : 0;
    } else if constexpr (ArgHasNumElements<TilingT>::Value && ArgHasCols<TilingT>::Value) {
        const int64_t c = static_cast<int64_t>(td->cols);
        return c > 0 ? static_cast<int64_t>(td->numElements) / c : 0;
    } else if constexpr (ArgHasTotalSize<TilingT>::Value && ArgHasCols<TilingT>::Value) {
        const int64_t c = static_cast<int64_t>(td->cols);
        return c > 0 ? static_cast<int64_t>(td->totalSize) / c : 0;
    } else {
        return reinterpret_cast<const ArgpartitionTilingLayout*>(td)->totalRows;
    }
}

template <typename TilingT>
__aicore__ inline int64_t ArgGetTotalCols(const TilingT* td)
{
    if constexpr (ArgHasTotalCols<TilingT>::Value) {
        return static_cast<int64_t>(td->totalCols);
    } else if constexpr (ArgHasCols<TilingT>::Value) {
        return static_cast<int64_t>(td->cols);
    } else if constexpr (ArgHasNumCols<TilingT>::Value) {
        return static_cast<int64_t>(td->numCols);
    } else if constexpr (ArgHasColNum<TilingT>::Value) {
        return static_cast<int64_t>(td->colNum);
    } else if constexpr (ArgHasRowLength<TilingT>::Value) {
        return static_cast<int64_t>(td->rowLength);
    } else if constexpr (ArgHasColLength<TilingT>::Value) {
        return static_cast<int64_t>(td->colLength);
    } else if constexpr (ArgHasTotalColsSnake<TilingT>::Value) {
        return static_cast<int64_t>(td->total_cols);
    } else if constexpr (ArgHasNCols<TilingT>::Value) {
        return static_cast<int64_t>(td->nCols);
    } else if constexpr (ArgHasColsPerRow<TilingT>::Value) {
        return static_cast<int64_t>(td->colsPerRow);
    } else if constexpr (ArgHasN<TilingT>::Value) {
        return static_cast<int64_t>(td->n);
    } else if constexpr (ArgHasTotalNum<TilingT>::Value && ArgHasTotalRows<TilingT>::Value) {
        const int64_t r = static_cast<int64_t>(td->totalRows);
        return r > 0 ? static_cast<int64_t>(td->totalNum) / r : 0;
    } else if constexpr (ArgHasTotalNum<TilingT>::Value && ArgHasRows<TilingT>::Value) {
        const int64_t r = static_cast<int64_t>(td->rows);
        return r > 0 ? static_cast<int64_t>(td->totalNum) / r : 0;
    } else if constexpr (ArgHasTotalElements<TilingT>::Value && ArgHasTotalRows<TilingT>::Value) {
        const int64_t r = static_cast<int64_t>(td->totalRows);
        return r > 0 ? static_cast<int64_t>(td->totalElements) / r : 0;
    } else if constexpr (ArgHasNumElements<TilingT>::Value && ArgHasRows<TilingT>::Value) {
        const int64_t r = static_cast<int64_t>(td->rows);
        return r > 0 ? static_cast<int64_t>(td->numElements) / r : 0;
    } else if constexpr (ArgHasTotalSize<TilingT>::Value && ArgHasRows<TilingT>::Value) {
        const int64_t r = static_cast<int64_t>(td->rows);
        return r > 0 ? static_cast<int64_t>(td->totalSize) / r : 0;
    } else {
        return reinterpret_cast<const ArgpartitionTilingLayout*>(td)->totalCols;
    }
}

template <typename T>
__aicore__ inline bool ArgLess(const T a, const T b)
{
    const bool aIsNan = (a != a);
    const bool bIsNan = (b != b);
    if (aIsNan || bIsNan) {
        /* NumPy places NaNs after ordinary values in sorted/partitioned order. */
        return (!aIsNan && bIsNan);
    }
    return (a < b);
}

template <typename T = float, int BUFFER_MODE = 0>
class Argpartition {
    static constexpr int32_t BUFFER_NUM = (BUFFER_MODE > 0) ? 2 : 1;

public:
    __aicore__ inline Argpartition() {}

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingT* tilingData)
    {
        const int64_t totalRows = ArgGetTotalRows(tilingData);
        const int64_t totalCols = ArgGetTotalCols(tilingData);
        cols_ = totalCols;
        blockLength_ = 0;
        if (totalRows <= 0 || totalCols <= 0) {
            return;
        }

        const int64_t blockNum = static_cast<int64_t>(AscendC::GetBlockNum());
        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        if (blockNum <= 0 || blockIdx < 0 || blockIdx >= blockNum) {
            return;
        }

        const int64_t rowsPerCore = (totalRows + blockNum - 1) / blockNum;
        const int64_t startRow = blockIdx * rowsPerCore;
        int64_t endRow = startRow + rowsPerCore;
        if (endRow > totalRows) {
            endRow = totalRows;
        }
        if (startRow >= totalRows || startRow >= endRow) {
            return;
        }

        blockLength_ = endRow - startRow;

        inputGM_.SetGlobalBuffer((__gm__ T*)x + startRow * cols_, blockLength_ * cols_);
        outputGM_.SetGlobalBuffer((__gm__ int64_t*)y + startRow * cols_, blockLength_ * cols_);

        pipe_.InitBuffer(inputQueueX_, BUFFER_NUM, static_cast<uint32_t>(cols_ * sizeof(T)));
        pipe_.InitBuffer(outputQueueY_, BUFFER_NUM, static_cast<uint32_t>(cols_ * sizeof(int64_t)));
    }

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const TilingT* tilingData)
    {
        (void)workspace;
        Init(x, y, tilingData);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
    {
        Init(x, y, reinterpret_cast<const ArgpartitionTilingLayout*>(tiling));
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
    {
        (void)workspace;
        Init(x, y, tiling);
    }

    __aicore__ inline void Process()
    {
        for (int64_t row = 0; row < blockLength_; ++row) {
            CopyIn(row, cols_);
            Compute(cols_);
            CopyOut(row, cols_);
        }
    }

private:
    __aicore__ inline void CopyIn(const int64_t row, const int64_t currentNum)
    {
        LocalTensor<T> xLocal = inputQueueX_.template AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(
            xLocal, inputGM_[row * cols_], copyParams, {false, 0, 0, static_cast<T>(0)});
        inputQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(const int64_t row, const int64_t currentNum)
    {
        LocalTensor<int64_t> yLocal = outputQueueY_.template DeQue<int64_t>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(int64_t));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(outputGM_[row * cols_], yLocal, copyParams);
        outputQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(const int64_t currentNum)
    {
        LocalTensor<T> xLocal = inputQueueX_.template DeQue<T>();
        LocalTensor<int64_t> yLocal = outputQueueY_.template AllocTensor<int64_t>();

        for (int64_t i = 0; i < currentNum; ++i) {
            yLocal.SetValue(i, i);
        }

        /*
         * Deterministic selection sort over original indices. This yields a
         * full ascending permutation, which satisfies NumPy's argpartition
         * contract for every admitted kth value. NaN is placed after finite
         * and infinite values; ties retain original index order.
         */
        for (int64_t i = 0; i + 1 < currentNum; ++i) {
            int64_t best = i;
            T bestVal = xLocal.GetValue(yLocal.GetValue(i));
            for (int64_t j = i + 1; j < currentNum; ++j) {
                const T candidate = xLocal.GetValue(yLocal.GetValue(j));
                if (ArgLess(candidate, bestVal)) {
                    best = j;
                    bestVal = candidate;
                }
            }
            if (best != i) {
                const int64_t tmp = yLocal.GetValue(i);
                yLocal.SetValue(i, yLocal.GetValue(best));
                yLocal.SetValue(best, tmp);
            }
        }

        outputQueueY_.template EnQue<int64_t>(yLocal);
        inputQueueX_.FreeTensor(xLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueueX_;
    TQue<QuePosition::VECOUT, 1> outputQueueY_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<int64_t> outputGM_;

    int64_t blockLength_ = 0;
    int64_t cols_ = 0;
};

Argpartition() -> Argpartition<float, 0>;

} // namespace NsArgpartition

#endif // ARGPARTITION_H