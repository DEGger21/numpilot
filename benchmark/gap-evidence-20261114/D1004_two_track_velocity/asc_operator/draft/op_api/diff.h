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
 * @file diff.h
 * @brief ACLNN L0 API declaration for Diff.
 *
 * Diff computes the first discrete difference of a float32 tensor along the
 * requested axis.  It is the L0 implementation called by aclnnDiff after the
 * L2 layer has validated parameters and made the input contiguous.
 */

#ifndef OP_API_INC_LEVEL0_DIFF_H_
#define OP_API_INC_LEVEL0_DIFF_H_

#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Compute the first forward difference along axis.
 *
 * For an input of shape S, the result has the same rank as S and its axis
 * dimension is reduced by one.  The input must be float32; unsupported
 * dtypes and invalid axes are rejected by the ACLNN parameter checks.
 *
 * @param input [in] Input tensor.
 * @param axis [in] Normalized non-negative axis along which to difference.
 * @param executor [in] ACLNN operator executor.
 * @return The allocated output tensor, or nullptr on failure.
 */
const aclTensor* Diff(const aclTensor* input, int64_t axis, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_DIFF_H_