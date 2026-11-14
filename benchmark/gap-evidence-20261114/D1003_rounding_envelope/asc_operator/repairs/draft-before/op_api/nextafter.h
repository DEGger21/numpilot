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
 * @file nextafter.h
 * @brief ACLNN L0 API declaration for the Nextafter operator.
 *
 * L0 API responsibilities:
 *   - infer the output shape from the two input tensors;
 *   - verify that the platform and dtype are supported;
 *   - allocate the result tensor;
 *   - dispatch the Ascend C kernel through ADD_TO_LAUNCHER_LIST_AICORE.
 *
 * The public ACLNN L2 entry points (aclnnNextafterGetWorkspaceSize /
 * aclnnNextafter) are declared in aclnn_nextafter.h and implemented in
 * aclnn_nextafter.cpp. They call l0op::Nextafter declared below.
 *
 * Semantic contract:
 *   result = numpy.nextafter(x1, x2)
 *
 * Argument order is fixed by the NumPy oracle:
 *   nextafter(x1, x2) -> element-wise next representable value after x1
 *   in the direction of x2.
 *
 * This draft candidate is specialized for float32 inputs and float32 output.
 * Other dtypes are rejected by the L0/L2 validation paths; they are never
 * silently cast.
 */

#ifndef OP_API_INC_LEVEL0_NEXTAFTER_H_
#define OP_API_INC_LEVEL0_NEXTAFTER_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief L0 entry for element-wise nextafter.
 *
 * @param[in] x1  First input tensor, float32, same shape as x2.
 * @param[in] x2  Second input tensor, float32, same shape as x1.
 * @param[in] executor  Op executor used to allocate the result tensor and
 *                      append the kernel launch task.
 * @return Newly allocated output tensor on success; nullptr on failure.
 *
 * The returned tensor is owned by the executor and must not be freed by the
 * caller. The function validates shape equality, float32 dtype support, and
 * the target platform before dispatching the kernel.
 */
const aclTensor* Nextafter(const aclTensor* x1,
                           const aclTensor* x2,
                           aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_NEXTAFTER_H_