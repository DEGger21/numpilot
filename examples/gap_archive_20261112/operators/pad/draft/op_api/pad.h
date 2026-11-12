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
 * @file pad.h
 * @brief ACLNN L0 API declarations for the Pad operator.
 *
 * This level-0 interface implements the bounded default constant-padding
 * subset of numpy.pad used by the registry-invoke operator:
 *
 *   numpy.pad(input, pad_width, mode="constant")
 *
 * Supported inputs are float32 tensors. The pad width is supplied as a
 * flattened host-side integer array in NumPy axis order:
 *
 *   [before_axis0, after_axis0, before_axis1, after_axis1, ...]
 *
 * Only mode "constant" is supported. The constant fill value is the NumPy
 * default of 0.0. Unsupported dtypes, modes, ranks, layouts, or malformed
 * pad-width arrays are rejected by the caller with an ACLNN error; there is
 * no CPU fallback.
 */

#ifndef OP_API_INC_LEVEL0_PAD_H_
#define OP_API_INC_LEVEL0_PAD_H_

#include "opdev/common_types.h"
#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Infer the padded output shape, allocate the output tensor, and
 *        dispatch the Pad kernel.
 *
 * The function parameter order follows the public NumPy call order:
 * input tensor, pad width, mode, executor.
 *
 * @param self     Input tensor. Must be a float32, rank-N, C-contiguous tensor.
 * @param padWidth Flattened pad widths of length 2 * rank(self), in NumPy
 *                 axis order.
 * @param mode     Padding mode. Only "constant" is accepted.
 * @param executor ACLNN executor used for output allocation and kernel launch.
 * @return Allocated output tensor on success, or nullptr on validation,
 *         allocation, or launch failure.
 */
const aclTensor* Pad(const aclTensor* self,
                     const aclIntArray* padWidth,
                     const char* mode,
                     aclOpExecutor* executor);

/**
 * @brief Convenience overload for the NumPy default mode ("constant").
 */
inline const aclTensor* Pad(const aclTensor* self,
                            const aclIntArray* padWidth,
                            aclOpExecutor* executor)
{
    return Pad(self, padWidth, "constant", executor);
}

} // namespace l0op

#endif // OP_API_INC_LEVEL0_PAD_H_