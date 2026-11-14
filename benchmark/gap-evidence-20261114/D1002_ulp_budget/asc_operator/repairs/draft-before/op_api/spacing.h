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
 * @file spacing.h
 * @brief ACLNN L0 API declaration for the unary Spacing operator.
 *
 * NumPy semantic source: numpy.spacing.
 *
 * The supported profile for this draft is a single C-contiguous float32
 * input tensor. The L0 API computes the output shape from the input shape,
 * allocates one float32 output tensor, and dispatches the Ascend C kernel.
 * Callers that need layout conversion or unsupported-dtype rejection perform
 * those checks before reaching this interface.
 */

#ifndef OP_API_INC_LEVEL0_SPACING_H_
#define OP_API_INC_LEVEL0_SPACING_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Compute the spacing of float32 values.
 *
 * @param self [in] Input tensor. Supported dtype: DT_FLOAT.
 *                  Supported layout: C-contiguous, same shape as output.
 * @param executor [in] ACLNN op executor used for output allocation and
 *                      kernel launch registration.
 * @return Newly allocated output tensor with the same shape and dtype as
 *         @p self, or nullptr when validation, allocation, or launch
 *         registration fails.
 */
const aclTensor* Spacing(const aclTensor* self, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_SPACING_H_