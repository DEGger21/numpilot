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
 * \file argmax.h
 * \brief Argmax 算子 kernel 类定义（arch22 架构，Ascend910B3）
 *
 * 语义与 numpy.argmax 严格对齐（accuracy-first，本阶段不做性能优化）：
 *   - 沿规约轴返回最大值**首次出现**的下标，输出 dtype 为 int64；
 *   - 并列最大值取首次出现（只有严格大于才更新）；
 *   - NaN 视为最大值；存在多个 NaN 时取首次出现的 NaN
 *     （等价于 NumPy 的 `x > best || (isnan(x) && !isnan(best))` 更新规则）；
 *   - ±0.0 按 IEEE 比较（-0.0 == +0.0，不更新下标），与 NumPy 一致；
 *   - ±inf 参与普通比较。
 *
 * 输入按 (outer, reduceLen, inner) 三分解读：
 *   outer     = prod(shape[0 : axis])          规约轴之前的元素总数
 *   reduceLen = shape[axis]                    规约轴长度
 *   inner     = prod(shape[axis + 1 : ])       规约轴之后的元素总数
 * 输出展平长度 = outer * inner，第 o 个输出元素对应
 *   row = o / inner, lane = o % inner,
 *   输入下标序列 = row * reduceLen * inner + lane + r * inner  (r = 0 .. reduceLen-1)
 * 已观测 profile（float32 / [1024, 1024] / axis=1 / 输出 int64 [1024]）即
 * outer=1024, reduceLen=1024, inner=1 的特化。范围外的 dtype / shape 由 host 侧
 * 显式拒绝，本 kernel 不做任何隐式 dtype 转换，也不做 CPU 回退。
 *
 * 执行结构：Init -> (CopyIn -> Compute)* -> CopyOut，每个输出元素独立规约，
 * 规约轴按 ubFactor 分块搬入 UB，块内以标量比较更新 (bestVal_, bestIdx_)。
 *
 * 模板参数说明（与 argmax_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 对应）：
 *   - T:           输入数据类型，本候选仅实例化 float32；
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲，默认；1=双缓冲队列深度），
 *                  默认为 0，因此 `Argmax<T>` 与 `Argmax<T, BUFFER_MODE>` 两种
 *                  入口写法均可编译。
 */
#ifndef ARGMAX_H
#define ARGMAX_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "argmax_tiling_data.h"

namespace NsArgmax {

using namespace AscendC;

// UB 搬运单元对齐字节数（DataCopyPad / InitBuffer 的最小对齐要求）
constexpr int64_t ARGMAX_UB_ALIGN_BYTES = 32;

__aicore__ inline int64_t ArgmaxAlignUpBytes(int64_t bytes)
{
    if (bytes <= 0) {
        return ARGMAX_UB_ALIGN_BYTES;
    }
    return (bytes + ARGMAX_UB_ALIGN_BYTES - 1) / ARGMAX_UB_ALIGN_BYTES * ARGMAX_UB_ALIGN_BYTES;
}

// 不依赖浮点比较指令的 NaN 判定，避免被有限数学优化抹掉（x != x 形式在
// 某些编译选项下不可靠）。
__aicore__ inline bool ArgmaxIsNanFloat(float value)
{
    uint32_t bits = 0U;
    float local = value;
    bits = *reinterpret_cast<uint32_t*>(&local);
    return ((bits & 0x7FFFFFFFU) > 0x7F800000U);
}

template <typename T, int BUFFER_MODE = 0>
class Argmax {
public:
    static constexpr int32_t BUFFER_NUM = (BUFFER_MODE == 0) ? 1 : 2;

    __aicore__ inline Argmax() {}

    // 三参数形式（不携带 workspace）与四参数形式同时提供，保证与 kernel 入口的调用方式一致。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR idx, const ArgmaxTilingData* tilingData);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR idx, GM_ADDR workspace, const ArgmaxTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t reduceBaseIndex, int64_t currentNum);
    __aicore__ inline void InitRunningValue();
    __aicore__ inline void Compute(int64_t reduceOffset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t outIndex);
    __aicore__ inline bool IsBetterThanBest(T value) const;
    __aicore__ inline bool IsNanValue(T value) const;

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<int64_t> outputGM_;

    int64_t outer_ = 0;        // 规约轴之前的元素总数
    int64_t reduceLen_ = 0;    // 规约轴长度
    int64_t inner_ = 0;        // 规约轴之后的元素总数
    int64_t totalOut_ = 0;     // outer_ * inner_
    int64_t blockFactor_ = 0;  // 每个核最多处理的输出元素数
    int64_t ubFactor_ = 0;     // 单次循环搬入的规约轴元素数
    int64_t outStart_ = 0;     // 当前核处理的起始输出下标
    int64_t blockLength_ = 0;  // 当前核实际处理的输出元素数

    T bestVal_ = static_cast<T>(0);
    int64_t bestIdx_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Argmax<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR idx, const ArgmaxTilingData* tilingData)
{
    Init(x, idx, nullptr, tilingData);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Argmax<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR idx, GM_ADDR workspace,
                                                    const ArgmaxTilingData* tilingData)
{
    // 本算子不需要额外 workspace，入口仍按固定参数顺序传入。
    (void)workspace;

    outer_ = 0;
    reduceLen_ = 0;
    inner_ = 0;
    totalOut_ = 0;
    blockFactor_ = 0;
    ubFactor_ = 1;
    outStart_ = 0;
    blockLength_ = 0;
    bestIdx_ = 0;
    bestVal_ = static_cast<T>(0);

    if (tilingData == nullptr) {
        return;
    }

    outer_ = tilingData->outerSize;
    reduceLen_ = tilingData->reduceSize;
    inner_ = tilingData->innerSize;
    totalOut_ = tilingData->totalOut;
    blockFactor_ = tilingData->blockFactor;
    ubFactor_ = (tilingData->ubFactor > 0) ? tilingData->ubFactor : 1;

    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(idx));

    // 空规约轴 / 非法 tiling：Process 直接返回，不产生任何读写。
    if (reduceLen_ <= 0 || inner_ <= 0 || totalOut_ <= 0 || blockFactor_ <= 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t start = blockIdx * blockFactor_;
    if (start >= totalOut_) {
        return;  // 尾核冗余核，无输出
    }
    const int64_t remain = totalOut_ - start;
    outStart_ = start;
    blockLength_ = (remain > blockFactor_) ? blockFactor_ : remain;

    const uint8_t bufferNum = static_cast<uint8_t>(BUFFER_NUM);
    const uint32_t inBytes =
        static_cast<uint32_t>(ArgmaxAlignUpBytes(ubFactor_ * static_cast<int64_t>(sizeof(T))));
    const uint32_t outBytes = static_cast<uint32_t>(ARGMAX_UB_ALIGN_BYTES);
    pipe_.InitBuffer(inputQueue_, bufferNum, inBytes);
    pipe_.InitBuffer(outputQueue_, bufferNum, outBytes);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Argmax<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    for (int64_t i = 0; i < blockLength_; ++i) {
        const int64_t outIndex = outStart_ + i;
        const int64_t row = outIndex / inner_;
        const int64_t lane = outIndex - row * inner_;
        const int64_t rowBase = row * reduceLen_ * inner_ + lane;

        // 第 0 个元素直接作为初始最大值，保证「并列取首次出现」。
        CopyIn(rowBase, 1);
        InitRunningValue();

        // 其余规约轴元素分块搬入并更新。
        for (int64_t r = 1; r < reduceLen_; r += ubFactor_) {
            const int64_t left = reduceLen_ - r;
            const int64_t currentNum = (left > ubFactor_) ? ubFactor_ : left;
            CopyIn(rowBase + r * inner_, currentNum);
            Compute(r, currentNum);
        }

        CopyOut(outIndex);
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Argmax<T, BUFFER_MODE>::CopyIn(int64_t reduceBaseIndex, int64_t currentNum)
{
    AscendC::LocalTensor<T> inLocal = inputQueue_.template AllocTensor<T>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    if (inner_ > 1) {
        // 规约轴非最内轴：按 inner 步长逐点收集，块数 = currentNum，块长 = 单元素字节。
        copyParams.blockCount = static_cast<uint16_t>(currentNum);
        copyParams.blockLen = static_cast<uint32_t>(sizeof(T));
        copyParams.srcStride = static_cast<uint32_t>((inner_ - 1) * static_cast<int64_t>(sizeof(T)));
        copyParams.dstStride = 0;
    }

    AscendC::DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = static_cast<T>(0);

    AscendC::DataCopyPad(inLocal, inputGM_[reduceBaseIndex], copyParams, padParams);
    inputQueue_.template EnQue<T>(inLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Argmax<T, BUFFER_MODE>::InitRunningValue()
{
    AscendC::LocalTensor<T> inLocal = inputQueue_.template DeQue<T>();
    bestVal_ = inLocal.GetValue(0);
    bestIdx_ = 0;
    inputQueue_.FreeTensor(inLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Argmax<T, BUFFER_MODE>::Compute(int64_t reduceOffset, int64_t currentNum)
{
    AscendC::LocalTensor<T> inLocal = inputQueue_.template DeQue<T>();
    for (int64_t i = 0; i < currentNum; ++i) {
        const T value = inLocal.GetValue(i);
        if (IsBetterThanBest(value)) {
            bestVal_ = value;
            bestIdx_ = reduceOffset + i;
        }
    }
    inputQueue_.FreeTensor(inLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Argmax<T, BUFFER_MODE>::CopyOut(int64_t outIndex)
{
    AscendC::LocalTensor<int64_t> outLocal = outputQueue_.template AllocTensor<int64_t>();
    outLocal.SetValue(0, bestIdx_);
    outputQueue_.template EnQue<int64_t>(outLocal);

    outLocal = outputQueue_.template DeQue<int64_t>();

    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(sizeof(int64_t));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    AscendC::DataCopyPad(outputGM_[outIndex], outLocal, copyParams);
    outputQueue_.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline bool Argmax<T, BUFFER_MODE>::IsBetterThanBest(T value) const
{
    // NumPy argmax 更新规则：严格大于才更新（并列取首次出现）；
    // NaN 视为最大值，且当前最优不是 NaN 时才接纳（首个 NaN 胜出）。
    if (value > bestVal_) {
        return true;
    }
    if (IsNanValue(value) && !IsNanValue(bestVal_)) {
        return true;
    }
    return false;
}

template <typename T, int BUFFER_MODE>
__aicore__ inline bool Argmax<T, BUFFER_MODE>::IsNanValue(T value) const
{
    if constexpr (std::is_same_v<T, float>) {
        return ArgmaxIsNanFloat(value);
    } else if constexpr (std::is_same_v<T, double>) {
        uint64_t bits = 0ULL;
        double local = value;
        bits = *reinterpret_cast<uint64_t*>(&local);
        return ((bits & 0x7FFFFFFFFFFFFFFFULL) > 0x7FF0000000000000ULL);
    } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        return ArgmaxIsNanFloat(static_cast<float>(value));
    } else {
        // 整型等非浮点类型不存在 NaN。
        return false;
    }
}

}  // namespace NsArgmax

#endif  // ARGMAX_H