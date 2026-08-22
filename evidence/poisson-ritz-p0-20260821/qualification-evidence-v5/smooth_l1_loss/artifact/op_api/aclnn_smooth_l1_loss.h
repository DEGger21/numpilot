/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_smooth_l1_loss.h
 * @brief ACLNN L2 API 接口声明 - SmoothL1Loss (mean reduction, beta=1)
 *
 * 语义（NumPy oracle）:
 *   error = predict - label
 *   absolute = |error|
 *   bounded = min(absolute, 1.0)
 *   out = mean(bounded^2 / 2 + (absolute - bounded))
 *
 * 输入: predict, label (float32, rank>=1, contiguous, 同形状)
 * 输出: out (float32 标量, shape=[1])
 *
 * 两段式调用:
 * - aclnnSmoothL1LossGetWorkspaceSize: 计算 workspace 大小，创建执行器
 * - aclnnSmoothL1Loss: 执行计算
 */

#ifndef ACLNN_SMOOTH_L1_LOSS_H_
#define ACLNN_SMOOTH_L1_LOSS_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnSmoothL1Loss 所需的 workspace 大小
 * @param predict [in] 输入张量 predict (float32)
 * @param label [in] 输入张量 label (float32)
 * @param out [in] 输出张量 (float32, shape=[1])
 * @param workspaceSize [out] 返回所需 workspace 大小
 * @param executor [out] 返回执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnSmoothL1LossGetWorkspaceSize(
    const aclTensor *predict,
    const aclTensor *label,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief 执行 SmoothL1Loss 算子计算
 * @param workspace [in] workspace 内存地址
 * @param workspaceSize [in] workspace 大小
 * @param executor [in] 执行器
 * @param stream [in] ACL 流
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnSmoothL1Loss(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SMOOTH_L1_LOSS_H_