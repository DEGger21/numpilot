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
 * @file std.h
 * @brief ACLNN L0 API interface for the Std operator.
 *
 * The L0 API performs output shape inference and schedules the Ascend C
 * kernel. The supported bounded surface for this candidate is one float32
 * input tensor, a scalar axis attribute, and one float32 output tensor.
 *
 * The implementation must normalize negative axes against the input rank,
 * reject axes outside [-rank, rank), and return nullptr on validation or
 * kernel-launch failure. No implicit dtype conversion or host fallback is
 * permitted.
 */

#ifndef OP_API_INC_LEVEL0_STD_H_
#define OP_API_INC_LEVEL0_STD_H_

#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Compute the standard deviation of a tensor over one axis.
 *
 * The output tensor is allocated by the L0 implementation using the input
 * dtype and the reduced shape. For a rank-N input and a valid normalized
 * axis, the output rank is N - 1. The reduction uses the same population
 * standard deviation definition as numpy.std with ddof=0.
 *
 * @param x [in] Input tensor. The current candidate accepts float32 only.
 * @param axis [in] Single reduction axis. Must be normalized by the callee.
 * @param executor [in] ACLNN operator executor used for allocation/launch.
 * @return Allocated output tensor on success, or nullptr on failure.
 */
const aclTensor* Std(const aclTensor* x, int64_t axis, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_STD_H_