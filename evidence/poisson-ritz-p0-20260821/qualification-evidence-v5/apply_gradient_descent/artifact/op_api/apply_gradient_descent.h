/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file apply_gradient_descent.h
 * @brief ACLNN L0 API 接口声明 - ApplyGradientDescent 算子
 *
 * L0 API 是 ACLNN 底层实现，由 L2 API（aclnn_apply_gradient_descent.cpp）调用。
 * 职责：形状推导、Kernel 调度。
 *
 * 语义（NumPy oracle，非原地）：
 *   out = var - alpha * delta
 *
 * 参数顺序与 NumPy oracle 一致：var, alpha, delta。
 * 支持类型：float32；形状规则：rank>=1，连续。
 */

#ifndef OP_API_INC_LEVEL0_APPLY_GRADIENT_DESCENT_H_
#define OP_API_INC_LEVEL0_APPLY_GRADIENT_DESCENT_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief ApplyGradientDescent L0 算子入口
 *
 * 流程：
 * 1. InferShape      - 形状推导（var 与 delta 广播，输出与广播结果同形）
 * 2. IsAiCoreSupport - 判断执行路径
 * 3. AllocTensor     - 分配输出 Tensor（dtype 与 var 一致：float32）
 * 4. ApplyGradientDescentAiCore - 调用 Kernel
 *
 * @param var [in] 输入张量 var（float32，rank>=1）
 * @param alpha [in] 标量/张量 alpha（float32）
 * @param delta [in] 输入张量 delta（float32，rank>=1）
 * @param executor [in] ACL 算子执行器
 * @return 输出张量（out = var - alpha * delta），失败返回 nullptr
 */
const aclTensor* ApplyGradientDescent(const aclTensor* var,
                                      const aclTensor* alpha,
                                      const aclTensor* delta,
                                      aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_APPLY_GRADIENT_DESCENT_H_