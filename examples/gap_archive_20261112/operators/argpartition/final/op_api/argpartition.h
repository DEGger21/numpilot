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
 * @file argpartition.h
 * @brief ACLNN L0 API declaration for the Argpartition operator.
 *
 * This level-0 entry point is called by the ACLNN L2 wrapper after the public
 * asnumpy.argpartition arguments have been validated and normalized.  The
 * implementation allocates a single int64 output tensor and schedules the
 * Argpartition kernel through its aclOpExecutor.
 */

#ifndef OP_API_INC_LEVEL0_ARGPARTITION_H_
#define OP_API_INC_LEVEL0_ARGPARTITION_H_

#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Infer the output shape/dtype and enqueue the Argpartition kernel.
 *
 * @param self [in] float32 input tensor.
 * @param kth [in] scalar partition index.
 * @param axis [in] normalized partition axis.
 * @param executor [in] ACLNN executor used for output allocation and launch.
 * @return the int64 output tensor, or nullptr on validation/launch failure.
 */
const aclTensor* Argpartition(const aclTensor* self,
                              int64_t kth,
                              int64_t axis,
                              aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_ARGPARTITION_H_