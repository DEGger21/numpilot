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
 * @file tril.h
 * @brief ACLNN L0 API declaration for the Tril operator.
 *
 * Public mapping:
 *   Python:  asnumpy.tril(a)
 *   C++ op:  Tril
 *   L2 API:  aclnnTrilGetWorkspaceSize / aclnnTril
 *   L0 API:  l0op::Tril
 *
 * The admitted source signature has exactly one positional input and no
 * keyword arguments.  The NumPy default diagonal k = 0 is therefore fixed in
 * this operator contract: elements with column index greater than row index
 * are zeroed, and the lower triangle plus diagonal is copied from the input.
 *
 * This draft stage targets the observed C-contiguous float32 rank-2 profile
 * only.  The L2 API owns argument validation and must explicitly reject other
 * dtypes, ranks, layouts, or signatures; it must not silently cast or fall
 * back to a host implementation.  The L0 entry below assumes those checks
 * have already succeeded.
 */

#ifndef OP_API_INC_LEVEL0_TRIL_H_
#define OP_API_INC_LEVEL0_TRIL_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Allocate and enqueue the Tril calculation.
 *
 * The output has the same shape and float32 dtype as the validated input.
 * The diagonal offset is fixed to zero to match the admitted asnumpy.tril
 * callsite signature.  Special values in the retained lower triangle are
 * copied without arithmetic reinterpretation.
 *
 * @param input [in] Validated C-contiguous float32 rank-2 input tensor.
 * @param executor [in,out] ACLNN executor used for output allocation and
 *        kernel launch registration.
 * @return Newly allocated output tensor, or nullptr on internal failure.
 */
const aclTensor* Tril(const aclTensor* input, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_TRIL_H_