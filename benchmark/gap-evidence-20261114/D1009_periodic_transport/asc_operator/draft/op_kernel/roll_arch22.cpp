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
 * \file roll_arch22.cpp
 * \brief Roll 算子 kernel 入口（Ascend910B3 / arch22，float32）
 *
 * 语义（NumPy oracle，唯一语义真值）：
 *   numpy.roll(x, shift, axis=axis)
 *   对 axis 维做周期搬移：y[..., i, ...] = x[..., (i - shift) mod dim(axis), ...]
 *   axis=None 时按 flatten 视图（总元素数）做一次周期搬移。
 *   Roll 是纯数据搬运算子：dtype / shape / layout / 特殊值（NaN、±Inf、±0 的符号位）全部保持。
 *
 * 本 kernel 的输入输出约定（与 op_host def / infershape / tiling、op_api L0/L2 必须一致）：
 *   - x      : 输入张量，float32，ND 且 C 连续（已观测 profile: [1024, 1024]）。
 *   - y      : 输出张量，float32，shape/dtype/layout 与 x 完全一致（单个 tensor 返回结构）。
 *   - shift  : int64 标量，Python 端第 2 个位置参数；不是 tensor 输入，经 host 归一化后进入 TilingData。
 *   - axis   : int64 标量，Python 端唯一关键字参数；不是 tensor 输入，经 host 归一化后折叠进 TilingData。
 *   kernel 入口参数顺序固定：输入 -> 输出 -> workspace -> tiling。
 *
 * kernel 依赖的 tiling 契约（op_host tiling 必须按此填写，二者字段名/语义必须逐一对应）：
 *   RollTilingData:
 *     totalNum     x/y 的元素总数
 *     rowLen       每行连续元素个数 = dim(axis) * inner，inner = prod(shape[axis+1:])
 *                  axis=None 时 rowLen = totalNum
 *     shiftOffset  行内右移元素个数 = (shift mod dim(axis)) * inner，取值 [0, rowLen)
 *                  axis=None 时 shiftOffset = shift mod totalNum
 *     rowCount     行数 = outer = prod(shape[:axis])；axis=None 时 rowCount = 1
 *     blockRows    每个核处理的行数（>=1；host 按 rowCount 与可用核数切分）
 *     ubFactor     单次 UB 搬运的元素个数（>=1，建议 32B 对齐，例如 8 的倍数）
 *
 * 关键等价变换（本 kernel 正确性的基础）：
 *   对 axis 维滚动后，"axis 及其之后的全部维度"构成一段长度为 rowLen = dim(axis) * inner 的
 *   连续内存，沿 axis 滚动 shift 步严格等价于对这一整段连续元素做循环右移 shiftOffset 个元素，
 *   即行内 dst[p] = src[(p - shiftOffset) mod rowLen]。
 *   因此每一行只需两段连续搬运（无需逐元素 gather，也不会在多维边界上出错）：
 *     dst[0, shiftOffset)            <- src[rowLen - shiftOffset, rowLen)
 *     dst[shiftOffset, rowLen)       <- src[0, rowLen - shiftOffset)
 *   该变换同时覆盖 axis=None（rowCount=1, rowLen=totalNum）与任意合法 axis / 任意 rank。
 *
 * 数值正确性说明：
 *   Compute 阶段使用 Muls(dst, src, 1.0f)：乘法 1.0 是 IEEE-754 精确恒等，NaN 保持 NaN、
 *   ±Inf 保持 ±Inf、-0.0 保持符号位（不采用 Adds(+0.0)，因为 -0.0 + 0.0 会翻转为 +0.0）。
 *   全程不做任何类型转换、不做饱和、不做舍入，输出与输入逐元素位级一致。
 *
 * 支持范围与拒绝范围（由 host 侧显式校验，kernel 不做隐式回退）：
 *   - 仅 float32 输入；其他 dtype 在 host / op_api 阶段显式拒绝，kernel 不做静默 cast。
 *   - 非法 axis、越界 axis、非连续输入由 host 归一化或显式拒绝。
 *   - 性能未在本阶段优化：按行、按 UB tile 做两段式搬运，正确性优先。
 */

#ifndef __ROLL_TILING_DATA_H__
#define __ROLL_TILING_DATA_H__

#include <cstdint>

/*!
 * \brief Roll 算子 tiling 数据结构（host 填写，kernel 读取）
 *
 * 注意：本文件会随 op_kernel/roll_arch22.cpp 一起编译；op_host 侧 tiling 必须使用
 * 字段名、字段顺序、字段类型完全一致的 RollTilingData 定义，否则 kernel 会读错切分参数。
 */
struct RollTilingData {
    int64_t totalNum = 0;    // x/y 元素总数
    int64_t rowLen = 0;      // 每行连续元素个数 = dim(axis) * inner
    int64_t shiftOffset = 0; // 行内右移元素个数 = (shift mod dim(axis)) * inner，范围 [0, rowLen)
    int64_t rowCount = 0;    // 行数 = outer（axis=None 时为 1）
    int64_t blockRows = 0;   // 每个核处理的行数
    int64_t ubFactor = 0;    // 单次 UB 搬运元素个数（>=1）
};

#endif // __ROLL_TILING_DATA_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace NsRoll {
using namespace AscendC;

/*! 单缓冲即可满足正确性；双缓冲属于性能优化，不在本阶段范围内。 */
constexpr int32_t ROLL_BUFFER_NUM = 1;

/*! UB 队列最小字节数（TQue::InitBuffer 的缓冲区长度按 32B 对齐申请）。 */
constexpr uint32_t ROLL_MIN_UB_BYTES = 32U;

/*!
 * \brief Roll kernel 主体：Init -> CopyIn -> Compute -> CopyOut
 *
 * 每个核负责行区间 [rowStart, rowEnd)，行内按 ubFactor 个元素分 tile 搬运。
 * 每个 tile 走一遍 GM -> UB (CopyIn) -> UB (Compute) -> GM (CopyOut) 的流水。
 */
template <typename T>
class Roll {
public:
    __aicore__ inline Roll() {}

    /*!
     * \brief 初始化 GM tensor、切分参数与 UB 队列
     * \param x 输入 tensor 的 GM 地址
     * \param y 输出 tensor 的 GM 地址
     * \param tilingData host 侧 tiling 数据
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RollTilingData* tilingData)
    {
        inGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), tilingData->totalNum);
        outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), tilingData->totalNum);

        rowLen_ = tilingData->rowLen;
        shiftOffset_ = tilingData->shiftOffset;
        rowCount_ = tilingData->rowCount;

        int64_t blockRows = tilingData->blockRows;
        if (blockRows <= 0) {
            // 防御：host 未给出有效并行切分时，退化为 0 号核处理全部行，保证结果完整正确
            blockRows = rowCount_;
        }

        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        rowStart_ = blockIdx * blockRows;
        if (rowStart_ > rowCount_) {
            rowStart_ = rowCount_;
        }
        rowEnd_ = rowStart_ + blockRows;
        if (rowEnd_ > rowCount_) {
            rowEnd_ = rowCount_;
        }

        ubFactor_ = tilingData->ubFactor;
        if (ubFactor_ <= 0) {
            ubFactor_ = 1;
        }

        uint32_t bufferBytes = static_cast<uint32_t>(ubFactor_ * static_cast<int64_t>(sizeof(T)));
        if (bufferBytes < ROLL_MIN_UB_BYTES) {
            bufferBytes = ROLL_MIN_UB_BYTES;
        }
        pipe_.InitBuffer(inQueue_, ROLL_BUFFER_NUM, bufferBytes);
        pipe_.InitBuffer(outQueue_, ROLL_BUFFER_NUM, bufferBytes);
    }

    /*!
     * \brief 主流程：逐行做两段式周期搬移
     */
    __aicore__ inline void Process()
    {
        if (rowCount_ <= 0 || rowLen_ <= 0) {
            return;
        }
        if (shiftOffset_ != 0) {
            // 防御：host 已归一化到 [0, rowLen)，这里再保证一次（含负位移）
            shiftOffset_ = ((shiftOffset_ % rowLen_) + rowLen_) % rowLen_;
        }
        if (shiftOffset_ >= rowLen_) {
            shiftOffset_ = 0;
        }

        // 行内 dst[p] = src[(p - shiftOffset) mod rowLen]
        //   段 1: dst[0, shiftOffset)         <- src[rowLen - shiftOffset, rowLen)
        //   段 2: dst[shiftOffset, rowLen)    <- src[0, rowLen - shiftOffset)
        const int64_t keepLen = rowLen_ - shiftOffset_;

        for (int64_t row = rowStart_; row < rowEnd_; ++row) {
            const int64_t rowBase = row * rowLen_;
            if (shiftOffset_ > 0) {
                CopySegment(rowBase + keepLen, rowBase, shiftOffset_);
            }
            if (keepLen > 0) {
                CopySegment(rowBase, rowBase + shiftOffset_, keepLen);
            }
        }
    }

private:
    /*!
     * \brief 搬运一段连续元素：src[srcOffset, srcOffset + len) -> dst[dstOffset, dstOffset + len)
     *        段内按 ubFactor 切片，最后一片允许非 32B 对齐（由 DataCopyPad 处理）。
     */
    __aicore__ inline void CopySegment(int64_t srcOffset, int64_t dstOffset, int64_t len)
    {
        int64_t done = 0;
        while (done < len) {
            int64_t curNum = len - done;
            if (curNum > ubFactor_) {
                curNum = ubFactor_;
            }
            CopyIn(srcOffset + done, curNum);
            Compute(curNum);
            CopyOut(dstOffset + done, curNum);
            done += curNum;
        }
    }

    /*! \brief GM -> UB，支持非 32B 对齐的起始地址与长度 */
    __aicore__ inline void CopyIn(int64_t srcOffset, int64_t count)
    {
        LocalTensor<T> inLocal = inQueue_.template AllocTensor<T>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(inLocal, inGM_[srcOffset], copyParams,
                             {false, 0, 0, static_cast<T>(0)});

        inQueue_.EnQue(inLocal);
    }

    /*! \brief Compute：UB->UB 原始数据搬移，保持 NaN payload / ±Inf / -0.0 位级不变 */
    __aicore__ inline void Compute(int64_t count)
    {
        LocalTensor<T> inLocal = inQueue_.template DeQue<T>();
        LocalTensor<T> outLocal = outQueue_.template AllocTensor<T>();

        // A pure permutation must not perform floating-point arithmetic: even
        // Muls(x, 1.0f) is allowed to canonicalise a NaN payload.  DataCopy
        // moves the raw 32B blocks and preserves every input bit.
        constexpr int64_t ELEM_ALIGN = static_cast<int64_t>(32 / sizeof(T));
        int64_t alignedNum = count;
        const int64_t remainder = alignedNum % ELEM_ALIGN;
        if (remainder != 0) {
            alignedNum += (ELEM_ALIGN - remainder);
        }
        if (alignedNum > ubFactor_) {
            alignedNum = ubFactor_;
        }
        AscendC::DataCopy(outLocal, inLocal, static_cast<uint32_t>(alignedNum));

        outQueue_.template EnQue<T>(outLocal);
        inQueue_.FreeTensor(inLocal);
    }

    /*! \brief UB -> GM，支持非 32B 对齐的目的地址与长度 */
    __aicore__ inline void CopyOut(int64_t dstOffset, int64_t count)
    {
        LocalTensor<T> outLocal = outQueue_.template DeQue<T>();

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(T)));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(outGM_[dstOffset], outLocal, copyParams);

        outQueue_.FreeTensor(outLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, ROLL_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECOUT, ROLL_BUFFER_NUM> outQueue_;

    GlobalTensor<T> inGM_;
    GlobalTensor<T> outGM_;

    int64_t ubFactor_ = 0;      // 单次 UB 搬运元素个数
    int64_t rowLen_ = 0;        // 每行连续元素个数
    int64_t shiftOffset_ = 0;   // 行内右移元素个数
    int64_t rowCount_ = 0;      // 总行数
    int64_t rowStart_ = 0;      // 本核负责的起始行（含）
    int64_t rowEnd_ = 0;        // 本核负责的结束行（不含）
};

} // namespace NsRoll

/*!
 * \brief Roll kernel 入口
 *
 * 参数顺序固定：输入 -> 输出 -> workspace -> tiling。
 * 本算子不使用 workspace（纯搬运、无跨核归约），但必须保留该形参以匹配 registry-invoke 调用约定。
 *
 * \param x          输入 tensor（float32，C 连续）
 * \param y          输出 tensor（float32，与 x 同 shape/dtype/layout）
 * \param workspace  系统预留 workspace 指针（未使用）
 * \param tiling     RollTilingData 序列化数据
 */
__global__ __aicore__ void roll(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RollTilingData);
    GET_TILING_DATA_WITH_STRUCT(RollTilingData, tilingData, tiling);

    if (tilingData.totalNum <= 0 || tilingData.rowCount <= 0 || tilingData.rowLen <= 0) {
        // 空张量：NumPy 返回同 shape 的空数组，无需任何搬运
        return;
    }

    NsRoll::Roll<float> op;
    op.Init(x, y, &tilingData);
    op.Process();
}