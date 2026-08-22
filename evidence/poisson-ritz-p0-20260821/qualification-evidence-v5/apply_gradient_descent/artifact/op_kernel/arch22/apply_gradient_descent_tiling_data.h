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
 * \file apply_gradient_descent_tiling_data.h
 * \brief ApplyGradientDescent 算子 tiling data 结构体（arch22 架构）
 *
 * 语义（NumPy oracle）：out = var - alpha * delta，逐元素计算。
 * 仅支持 float32，rank>=1 连续张量。
 */

#ifndef _APPLY_GRADIENT_DESCENT_TILING_DATA_H_
#define _APPLY_GRADIENT_DESCENT_TILING_DATA_H_

struct ApplyGradientDescentTilingData {
    int64_t totalNum = 0;     // 总元素数量（var/delta/output 元素个数一致）
    int64_t blockFactor = 0;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
};
#endif // _APPLY_GRADIENT_DESCENT_TILING_DATA_H_