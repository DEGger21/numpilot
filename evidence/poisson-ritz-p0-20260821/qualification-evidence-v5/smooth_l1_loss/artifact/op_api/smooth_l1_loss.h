/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

/**
 * @file smooth_l1_loss.h
 * @brief ACLNN L0 API 接口声明 - SmoothL1Loss (mean reduction, beta=1)
 *
 * 语义（NumPy oracle）:
 *   error = predict - label
 *   absolute = |error|
 *   bounded = min(absolute, 1.0)
 *   loss = mean(bounded^2 / 2 + (absolute - bounded))
 *
 * 输入: predict, label (float32, rank>=1, contiguous, 同形)
 * 输出: 标量 float32 (shape = [1])
 */

#ifndef OP_API_INC_LEVEL0_SMOOTH_L1_LOSS_H_
#define OP_API_INC_LEVEL0_SMOOTH_L1_LOSS_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief SmoothL1Loss L0 算子入口
 * @param predict [in] 预测值张量 (float32, rank>=1, contiguous)
 * @param label [in] 标签张量 (float32, 与 predict 同形)
 * @param executor [in] ACLNN 执行器
 * @return 输出标量张量 (float32, shape=[1])，失败返回 nullptr
 */
const aclTensor* SmoothL1Loss(const aclTensor* predict,
                             const aclTensor* label,
                             aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_SMOOTH_L1_LOSS_H_