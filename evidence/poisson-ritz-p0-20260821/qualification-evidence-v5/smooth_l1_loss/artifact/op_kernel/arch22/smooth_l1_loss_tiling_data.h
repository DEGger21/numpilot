/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_tiling_data.h
 * \brief SmoothL1Loss 算子 tiling data 结构体（arch22 / ascend910b）
 *
 * 语义（NumPy oracle，mean reduction，beta=1，float32）：
 *   error = predict - label
 *   absolute = |error|
 *   bounded = min(absolute, 1.0)
 *   loss_elem = bounded * bounded / 2.0 + (absolute - bounded)
 *   output = mean(loss_elem)   // 标量输出
 *
 * Kernel 采用两阶段策略：
 *   1) 各核分块计算本核元素损失的部分和（float32 累加）；
 *   2) 核 0 对各核部分和做最终归约并除以 totalNum 得到 mean。
 */

#ifndef _SMOOTH_L1_LOSS_TILING_DATA_H_
#define _SMOOTH_L1_LOSS_TILING_DATA_H_

struct SmoothL1LossTilingData {
    int64_t totalNum = 0;     // 输入总元素数量（predict/label 相同）
    int64_t blockFactor = 0;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
    int64_t coreNum = 0;      // 实际使用的核数（用于最终归约）
    int64_t finalReduce = 0;  // 是否由核 0 执行最终归约（1=是，0=否）
};
#endif // _SMOOTH_L1_LOSS_TILING_DATA_H_