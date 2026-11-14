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
 * \file diff.h
 * \brief Diff operator kernel class definition for arch22 (Ascend910B3).
 *
 * Bounded semantic scope for this draft:
 *   - float32 input
 *   - rank-2 C-contiguous input
 *   - numpy.diff(x, axis=1), first difference
 *   - output shape [rows, cols - 1]
 *
 * The host-side op definition and tiling must reject all other dtype, rank,
 * layout, order and axis combinations. This kernel intentionally assumes the
 * admitted axis=1, order=1 profile and computes y[r, c] = x[r, c + 1] - x[r, c].
 */

#ifndef NUMPILOT_DIFF_ARCH22_H
#define NUMPILOT_DIFF_ARCH22_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "diff_tiling_data.h"
#include "diff_tiling_key.h"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace NsDiff {

using namespace AscendC;

namespace TilingView {

#define NUMPILOT_DIFF_FIELD_TRAIT(TRAIT_NAME, FIELD_NAME)                         \
    template <typename TD, typename = void>                                       \
    struct TRAIT_NAME : std::false_type {};                                       \
    template <typename TD>                                                        \
    struct TRAIT_NAME<TD, std::void_t<decltype(std::declval<TD&>().FIELD_NAME)>> \
        : std::true_type {};

NUMPILOT_DIFF_FIELD_TRAIT(HasTotalRows, totalRows)
NUMPILOT_DIFF_FIELD_TRAIT(HasRows, rows)
NUMPILOT_DIFF_FIELD_TRAIT(HasRowNum, rowNum)
NUMPILOT_DIFF_FIELD_TRAIT(HasRowCount, rowCount)
NUMPILOT_DIFF_FIELD_TRAIT(HasNumRows, numRows)
NUMPILOT_DIFF_FIELD_TRAIT(HasInputRows, inputRows)
NUMPILOT_DIFF_FIELD_TRAIT(HasRowsNum, rowsNum)
NUMPILOT_DIFF_FIELD_TRAIT(HasM, m)
NUMPILOT_DIFF_FIELD_TRAIT(HasRow, row)
NUMPILOT_DIFF_FIELD_TRAIT(HasRowSize, rowSize)
NUMPILOT_DIFF_FIELD_TRAIT(HasHeight, height)
NUMPILOT_DIFF_FIELD_TRAIT(HasH, h)

NUMPILOT_DIFF_FIELD_TRAIT(HasInputCols, inputCols)
NUMPILOT_DIFF_FIELD_TRAIT(HasCols, cols)
NUMPILOT_DIFF_FIELD_TRAIT(HasColNum, colNum)
NUMPILOT_DIFF_FIELD_TRAIT(HasColCount, colCount)
NUMPILOT_DIFF_FIELD_TRAIT(HasNumCols, numCols)
NUMPILOT_DIFF_FIELD_TRAIT(HasColumns, columns)
NUMPILOT_DIFF_FIELD_TRAIT(HasN, n)
NUMPILOT_DIFF_FIELD_TRAIT(HasCol, col)
NUMPILOT_DIFF_FIELD_TRAIT(HasColSize, colSize)
NUMPILOT_DIFF_FIELD_TRAIT(HasWidth, width)
NUMPILOT_DIFF_FIELD_TRAIT(HasW, w)

NUMPILOT_DIFF_FIELD_TRAIT(HasOutCols, outCols)
NUMPILOT_DIFF_FIELD_TRAIT(HasOutputCols, outputCols)
NUMPILOT_DIFF_FIELD_TRAIT(HasOutColNum, outColNum)
NUMPILOT_DIFF_FIELD_TRAIT(HasOutColumns, outColumns)
NUMPILOT_DIFF_FIELD_TRAIT(HasNOut, nOut)
NUMPILOT_DIFF_FIELD_TRAIT(HasOutWidth, outWidth)
NUMPILOT_DIFF_FIELD_TRAIT(HasOutputWidth, outputWidth)
NUMPILOT_DIFF_FIELD_TRAIT(HasOutW, outW)

NUMPILOT_DIFF_FIELD_TRAIT(HasTotalNum, totalNum)
NUMPILOT_DIFF_FIELD_TRAIT(HasTotalElements, totalElements)
NUMPILOT_DIFF_FIELD_TRAIT(HasElementCount, elementCount)
NUMPILOT_DIFF_FIELD_TRAIT(HasNumElements, numElements)
NUMPILOT_DIFF_FIELD_TRAIT(HasTotalSize, totalSize)
NUMPILOT_DIFF_FIELD_TRAIT(HasTotalCount, totalCount)
NUMPILOT_DIFF_FIELD_TRAIT(HasElements, elements)

#undef NUMPILOT_DIFF_FIELD_TRAIT

template <typename TD>
__aicore__ inline int64_t GetRows(const TD* tilingData)
{
    if constexpr (HasTotalRows<TD>::value) {
        return static_cast<int64_t>(tilingData->totalRows);
    } else if constexpr (HasRows<TD>::value) {
        return static_cast<int64_t>(tilingData->rows);
    } else if constexpr (HasRowNum<TD>::value) {
        return static_cast<int64_t>(tilingData->rowNum);
    } else if constexpr (HasRowCount<TD>::value) {
        return static_cast<int64_t>(tilingData->rowCount);
    } else if constexpr (HasNumRows<TD>::value) {
        return static_cast<int64_t>(tilingData->numRows);
    } else if constexpr (HasInputRows<TD>::value) {
        return static_cast<int64_t>(tilingData->inputRows);
    } else if constexpr (HasRowsNum<TD>::value) {
        return static_cast<int64_t>(tilingData->rowsNum);
    } else if constexpr (HasM<TD>::value) {
        return static_cast<int64_t>(tilingData->m);
    } else if constexpr (HasRow<TD>::value) {
        return static_cast<int64_t>(tilingData->row);
    } else if constexpr (HasRowSize<TD>::value) {
        return static_cast<int64_t>(tilingData->rowSize);
    } else if constexpr (HasHeight<TD>::value) {
        return static_cast<int64_t>(tilingData->height);
    } else if constexpr (HasH<TD>::value) {
        return static_cast<int64_t>(tilingData->h);
    } else {
        return 0;
    }
}

template <typename TD>
__aicore__ inline int64_t GetCols(const TD* tilingData)
{
    if constexpr (HasInputCols<TD>::value) {
        return static_cast<int64_t>(tilingData->inputCols);
    } else if constexpr (HasCols<TD>::value) {
        return static_cast<int64_t>(tilingData->cols);
    } else if constexpr (HasColNum<TD>::value) {
        return static_cast<int64_t>(tilingData->colNum);
    } else if constexpr (HasColCount<TD>::value) {
        return static_cast<int64_t>(tilingData->colCount);
    } else if constexpr (HasNumCols<TD>::value) {
        return static_cast<int64_t>(tilingData->numCols);
    } else if constexpr (HasColumns<TD>::value) {
        return static_cast<int64_t>(tilingData->columns);
    } else if constexpr (HasN<TD>::value) {
        return static_cast<int64_t>(tilingData->n);
    } else if constexpr (HasCol<TD>::value) {
        return static_cast<int64_t>(tilingData->col);
    } else if constexpr (HasColSize<TD>::value) {
        return static_cast<int64_t>(tilingData->colSize);
    } else if constexpr (HasWidth<TD>::value) {
        return static_cast<int64_t>(tilingData->width);
    } else if constexpr (HasW<TD>::value) {
        return static_cast<int64_t>(tilingData->w);
    } else {
        return 0;
    }
}

template <typename TD>
__aicore__ inline int64_t GetOutCols(const TD* tilingData)
{
    if constexpr (HasOutCols<TD>::value) {
        return static_cast<int64_t>(tilingData->outCols);
    } else if constexpr (HasOutputCols<TD>::value) {
        return static_cast<int64_t>(tilingData->outputCols);
    } else if constexpr (HasOutColNum<TD>::value) {
        return static_cast<int64_t>(tilingData->outColNum);
    } else if constexpr (HasOutColumns<TD>::value) {
        return static_cast<int64_t>(tilingData->outColumns);
    } else if constexpr (HasNOut<TD>::value) {
        return static_cast<int64_t>(tilingData->nOut);
    } else if constexpr (HasOutWidth<TD>::value) {
        return static_cast<int64_t>(tilingData->outWidth);
    } else if constexpr (HasOutputWidth<TD>::value) {
        return static_cast<int64_t>(tilingData->outputWidth);
    } else if constexpr (HasOutW<TD>::value) {
        return static_cast<int64_t>(tilingData->outW);
    } else {
        return 0;
    }
}

template <typename TD>
__aicore__ inline int64_t GetTotalNum(const TD* tilingData)
{
    if constexpr (HasTotalNum<TD>::value) {
        return static_cast<int64_t>(tilingData->totalNum);
    } else if constexpr (HasTotalElements<TD>::value) {
        return static_cast<int64_t>(tilingData->totalElements);
    } else if constexpr (HasElementCount<TD>::value) {
        return static_cast<int64_t>(tilingData->elementCount);
    } else if constexpr (HasNumElements<TD>::value) {
        return static_cast<int64_t>(tilingData->numElements);
    } else if constexpr (HasTotalSize<TD>::value) {
        return static_cast<int64_t>(tilingData->totalSize);
    } else if constexpr (HasTotalCount<TD>::value) {
        return static_cast<int64_t>(tilingData->totalCount);
    } else if constexpr (HasElements<TD>::value) {
        return static_cast<int64_t>(tilingData->elements);
    } else {
        return 0;
    }
}

} // namespace TilingView

template <typename T = float, int BUFFER_MODE = 1>
class Diff {
public:
    __aicore__ inline Diff()
        : valid_(false), rows_(0), cols_(1), outCols_(0), startRow_(0), endRow_(0)
    {
    }

    template <typename TD>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TD* tilingData);

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const DiffTilingData& tilingData)
    {
        Init(x, y, &tilingData);
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t row);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int64_t row);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue0_;
    TQue<QuePosition::VECIN, 1> inputQueue1_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    bool valid_;
    int64_t rows_;
    int64_t cols_;
    int64_t outCols_;
    int64_t startRow_;
    int64_t endRow_;
};

template <typename T, int BUFFER_MODE>
template <typename TD>
__aicore__ inline void Diff<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y, const TD* tilingData)
{
    valid_ = false;
    rows_ = 0;
    cols_ = 1;
    outCols_ = 0;
    startRow_ = 0;
    endRow_ = 0;

    if (tilingData == nullptr) {
        return;
    }

    rows_ = TilingView::GetRows(tilingData);
    cols_ = TilingView::GetCols(tilingData);
    const int64_t totalNum = TilingView::GetTotalNum(tilingData);
    const int64_t outColsHint = TilingView::GetOutCols(tilingData);

    if (cols_ <= 1 && outColsHint > 0) {
        cols_ = outColsHint + 1;
    }

    if (rows_ <= 0 && totalNum > 0 && cols_ > 1) {
        if ((totalNum % cols_) == 0) {
            rows_ = totalNum / cols_;
        } else if ((totalNum % (cols_ - 1)) == 0) {
            rows_ = totalNum / (cols_ - 1);
        }
    }

    if (rows_ <= 0 && totalNum > 0 && outColsHint > 0) {
        if ((totalNum % outColsHint) == 0) {
            rows_ = totalNum / outColsHint;
        } else if ((totalNum % (outColsHint + 1)) == 0) {
            rows_ = totalNum / (outColsHint + 1);
        }
    }

    if (cols_ <= 1 && totalNum > 0 && rows_ > 0) {
        if ((totalNum % rows_) == 0) {
            const int64_t candidate = totalNum / rows_;
            if (candidate > 1) {
                cols_ = candidate;
            }
        }
        if (cols_ <= 1) {
            const int64_t candidate = totalNum / rows_ + 1;
            if (candidate > 1) {
                cols_ = candidate;
            }
        }
    }

    if (rows_ <= 0 || cols_ <= 1) {
        return;
    }

    outCols_ = cols_ - 1;

    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockNum <= 0) {
        blockNum = 1;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t rowsPerCore = (rows_ + blockNum - 1) / blockNum;
    startRow_ = blockIdx * rowsPerCore;
    if (startRow_ >= rows_) {
        endRow_ = startRow_;
        return;
    }

    endRow_ = startRow_ + rowsPerCore;
    if (endRow_ > rows_) {
        endRow_ = rows_;
    }

    inputGM_.SetGlobalBuffer((__gm__ T*)x, rows_ * cols_);
    outputGM_.SetGlobalBuffer((__gm__ T*)y, rows_ * outCols_);

    const int64_t alignedBytes =
        ((outCols_ * static_cast<int64_t>(sizeof(T)) + 31) / 32) * 32;
    const uint32_t bufferBytes = static_cast<uint32_t>(alignedBytes);
    pipe_.InitBuffer(inputQueue0_, 1, bufferBytes);
    pipe_.InitBuffer(inputQueue1_, 1, bufferBytes);
    pipe_.InitBuffer(outputQueue_, 1, bufferBytes);

    valid_ = true;
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Diff<T, BUFFER_MODE>::CopyIn(int64_t row)
{
    LocalTensor<T> x0 = inputQueue0_.template AllocTensor<T>();
    LocalTensor<T> x1 = inputQueue1_.template AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(outCols_ * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPad(x0, inputGM_[row * cols_], copyParams, {false, 0, 0, static_cast<T>(0)});
    DataCopyPad(x1, inputGM_[row * cols_ + 1], copyParams, {false, 0, 0, static_cast<T>(0)});

    inputQueue0_.EnQue(x0);
    inputQueue1_.EnQue(x1);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Diff<T, BUFFER_MODE>::Compute()
{
    LocalTensor<T> x0 = inputQueue0_.template DeQue<T>();
    LocalTensor<T> x1 = inputQueue1_.template DeQue<T>();
    LocalTensor<T> y = outputQueue_.template AllocTensor<T>();

    Sub(y, x1, x0, static_cast<uint32_t>(outCols_));

    outputQueue_.template EnQue<T>(y);
    inputQueue0_.FreeTensor(x0);
    inputQueue1_.FreeTensor(x1);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Diff<T, BUFFER_MODE>::CopyOut(int64_t row)
{
    LocalTensor<T> y = outputQueue_.template DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(outCols_ * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPad(outputGM_[row * outCols_], y, copyParams);
    outputQueue_.FreeTensor(y);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Diff<T, BUFFER_MODE>::Process()
{
    if (!valid_) {
        return;
    }

    for (int64_t row = startRow_; row < endRow_; ++row) {
        CopyIn(row);
        Compute();
        CopyOut(row);
    }
}

} // namespace NsDiff

#endif // NUMPILOT_DIFF_ARCH22_H