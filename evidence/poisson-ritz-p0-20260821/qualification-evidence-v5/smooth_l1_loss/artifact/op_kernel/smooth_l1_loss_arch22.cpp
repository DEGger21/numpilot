/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_arch22.cpp
 * \brief SmoothL1Loss kernel entry (arch22, ascend910b)
 *
 * Internal OpType: NumPilotSmoothMeanLoss (digit-free so that CANN 9.0's
 * build-time and runtime CamelCase-to-file-name conversions agree; the
 * previous NumPilotSmoothL1Loss embedded digit "1" caused the runtime to
 * look for num_pilot_smooth_l_1_loss.json / num_pilot_smooth_l1loss.json
 * while the build installed num_pilot_smooth_l1_loss.json, producing
 * ACLNN 561103 / 561108). The public ACLNN symbols aclnnSmoothL1Loss* and
 * the Python API asnumpy.smooth_l1_loss are unchanged.
 *
 * Semantics (NumPy oracle, mean / beta=1, float32):
 *   error     = predict - label
 *   absolute  = |error|
 *   bounded   = min(absolute, 1.0f)
 *   quadratic = bounded * bounded * 0.5f
 *   linear    = absolute - bounded
 *   output    = mean(linear + quadratic)   // scalar
 *
 * Strategy: single AIV core (blockDim=1, blockFactor=totalNum set by host
 * tiling). The kernel accumulates the per-element loss over UB tiles, reduces
 * to a scalar, divides by totalNum and writes the scalar directly. No atomic
 * add is used.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch22/smooth_l1_loss_tiling_data.h"

namespace NsSmoothL1Loss {

using namespace AscendC;

class SmoothL1LossKernel {
public:
    __aicore__ inline SmoothL1LossKernel() {}

    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR loss,
                                const SmoothL1LossTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> quePredict_;
    TQue<QuePosition::VECIN, 1> queLabel_;

    TBuf<TPosition::VECCALC> absoluteBuf_;
    TBuf<TPosition::VECCALC> boundedBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> outBuf_;

    GlobalTensor<float> gmPredict_;
    GlobalTensor<float> gmLabel_;
    GlobalTensor<float> gmLoss_;

    int64_t blockLength_ = 0;  // elements handled by this core
    int64_t ubLength_ = 0;     // elements per UB loop
    int64_t totalNum_ = 0;     // global element count (for mean)
    float blockSum_ = 0.0f;    // accumulated loss sum on this core
};

__aicore__ inline void SmoothL1LossKernel::Init(GM_ADDR predict, GM_ADDR label, GM_ADDR loss,
                                                const SmoothL1LossTilingData *tilingData)
{
    totalNum_ = tilingData->totalNum;
    int64_t remainderLength = totalNum_ - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    gmPredict_.SetGlobalBuffer((__gm__ float *)predict + tilingData->blockFactor * AscendC::GetBlockIdx(),
                               blockLength_);
    gmLabel_.SetGlobalBuffer((__gm__ float *)label + tilingData->blockFactor * AscendC::GetBlockIdx(),
                             blockLength_);
    gmLoss_.SetGlobalBuffer((__gm__ float *)loss, 1);

    pipe.InitBuffer(quePredict_, 1, ubLength_ * sizeof(float));
    pipe.InitBuffer(queLabel_, 1, ubLength_ * sizeof(float));
    pipe.InitBuffer(absoluteBuf_, ubLength_ * sizeof(float));
    pipe.InitBuffer(boundedBuf_, ubLength_ * sizeof(float));
    pipe.InitBuffer(reduceBuf_, ubLength_ * sizeof(float));
    // separate 32-byte output-local buffer
    pipe.InitBuffer(outBuf_, 32);
}

__aicore__ inline void SmoothL1LossKernel::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<float> predictLocal = quePredict_.AllocTensor<float>();
    LocalTensor<float> labelLocal = queLabel_.AllocTensor<float>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(predictLocal, gmPredict_[progress * ubLength_], copyParams,
                         {false, 0, 0, static_cast<float>(0)});
    AscendC::DataCopyPad(labelLocal, gmLabel_[progress * ubLength_], copyParams,
                         {false, 0, 0, static_cast<float>(0)});
    quePredict_.EnQue(predictLocal);
    queLabel_.EnQue(labelLocal);
}

__aicore__ inline void SmoothL1LossKernel::Compute(int64_t currentNum)
{
    LocalTensor<float> predictLocal = quePredict_.DeQue<float>();
    LocalTensor<float> labelLocal = queLabel_.DeQue<float>();

    LocalTensor<float> absoluteLocal = absoluteBuf_.Get<float>();
    LocalTensor<float> boundedLocal = boundedBuf_.Get<float>();
    LocalTensor<float> reduceDstLocal = reduceBuf_.Get<float>();

    // error = predict - label ; absolute = |error|
    AscendC::Sub(absoluteLocal, predictLocal, labelLocal, currentNum);
    AscendC::Abs(absoluteLocal, absoluteLocal, currentNum);

    // bounded = min(absolute, 1.0f)
    AscendC::Mins(boundedLocal, absoluteLocal, static_cast<float>(1.0f), currentNum);
    // linear = absolute - bounded (must use the original bounded value)
    AscendC::Sub(absoluteLocal, absoluteLocal, boundedLocal, currentNum); // linear
    // quadratic = bounded^2 * 0.5f
    AscendC::Mul(boundedLocal, boundedLocal, boundedLocal, currentNum);   // square
    AscendC::Muls(boundedLocal, boundedLocal, static_cast<float>(0.5f), currentNum); // quadratic
    // elementLoss = linear + quadratic
    AscendC::Add(reduceDstLocal, absoluteLocal, boundedLocal, currentNum);

    // reduce this tile to a scalar and accumulate; ReduceSum returns void,
    // so call it as a statement and read element zero of the destination.
    AscendC::ReduceSum(reduceDstLocal, reduceDstLocal, absoluteLocal, currentNum);
    blockSum_ += reduceDstLocal.GetValue(0);

    quePredict_.FreeTensor(predictLocal);
    queLabel_.FreeTensor(labelLocal);
}

__aicore__ inline void SmoothL1LossKernel::CopyOut()
{
    if (blockLength_ <= 0) {
        return;
    }
    LocalTensor<float> outLocal = outBuf_.Get<float>();
    float mean = blockSum_ / static_cast<float>(totalNum_);
    AscendC::Duplicate(outLocal, mean, 8); // 8 floats = 32B aligned, first element valid
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(float); // exactly one float
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(gmLoss_, outLocal, copyParams);
}

__aicore__ inline void SmoothL1LossKernel::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
    }
    CopyOut();
}

} // namespace NsSmoothL1Loss

__global__ __aicore__ void num_pilot_smooth_mean_loss(GM_ADDR predict, GM_ADDR label, GM_ADDR loss,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SmoothL1LossTilingData);
    GET_TILING_DATA_WITH_STRUCT(SmoothL1LossTilingData, tilingData, tiling);
    NsSmoothL1Loss::SmoothL1LossKernel op;
    op.Init(predict, label, loss, &tilingData);
    op.Process();
}