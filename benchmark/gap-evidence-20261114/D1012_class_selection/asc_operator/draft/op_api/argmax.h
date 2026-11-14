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
 * @file argmax.h
 * @brief ACLNN L0 API interface declaration for the Argmax operator.
 *
 * The L0 API is the internal ACLNN implementation invoked by the L2 API
 * (op_api/aclnn_argmax.h). It performs output shape inference and dispatches
 * the Argmax kernel through the supplied op executor.
 *
 * Public mapping:
 *   asnumpy.argmax -> Python wrapper -> pybind -> native adapter
 *   -> aclnnArgmaxGetWorkspaceSize / aclnnArgmax
 *   -> l0op::Argmax
 */

#ifndef OP_API_INC_LEVEL0_ARGMAX_H_
#define OP_API_INC_LEVEL0_ARGMAX_H_

#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Argmax L0 entry point.
 *
 * Semantics follow numpy.argmax(input, axis=axis): return the indices of the
 * maximum values along the requested axis. The output is int64 and its rank is
 * input rank - 1. This draft stage is scoped to float32 C-contiguous inputs;
 * unsupported dtypes must be rejected by the callers rather than silently
 * cast.
 *
 * @param input    [in] contiguous input tensor (float32 supported in this draft)
 * @param axis     [in] reduction axis, interpreted with NumPy axis normalization
 * @param executor [in] ACLNN op executor used for allocation and kernel launch
 * @return int64 output tensor on success, nullptr on failure
 */
const aclTensor* Argmax(const aclTensor* input, int64_t axis, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_ARGMAX_H_