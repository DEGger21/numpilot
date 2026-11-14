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
 * @file roll.h
 * @brief ACLNN L0 API interface declaration for Roll.
 *
 * Public mapping:
 *   asnumpy.roll(self, shift, axis=...)
 *     -> Python adapter
 *     -> pybind entry
 *     -> L2 ACLNN aclnnRollGetWorkspaceSize / aclnnRoll
 *     -> L0 ACLNN l0op::Roll
 *
 * The L0 API allocates and returns the output tensor. It does not perform
 * Python-level argument handling; the caller must pass already-normalized
 * scalar shift and axis attributes. The supported candidate profile is a
 * float32, rank-2, C-contiguous input with axis=1, matching the observed
 * NumPy call site. Other dtypes and profiles are rejected by the L2/L0
 * parameter checks rather than silently converted.
 */

#ifndef OP_API_INC_LEVEL0_ROLL_H_
#define OP_API_INC_LEVEL0_ROLL_H_

#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Apply NumPy-compatible roll to the selected axis.
 * @param self [in] Input tensor. The candidate qualification profile is float32,
 *                  rank 2, C-contiguous.
 * @param shift [in] Number of positions to shift. Positive and negative values
 *                   follow numpy.roll semantics after modulo normalization on
 *                   the selected axis.
 * @param axis [in] Axis on which to roll. The observed supported candidate
 *                  profile uses axis=1.
 * @param executor [in] ACLNN operator executor used for output allocation and
 *                      kernel launch registration.
 * @return Newly allocated output tensor with the same dtype and shape as self,
 *         or nullptr when shape/dtype validation or launch registration fails.
 */
const aclTensor* Roll(const aclTensor* self,
                      int64_t shift,
                      int64_t axis,
                      aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_ROLL_H_