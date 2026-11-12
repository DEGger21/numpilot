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
 * @file flip.h
 * @brief ACLNN L0 API declaration for Flip.
 *
 * Supported draft contract:
 * - input dtype: float32 only;
 * - output dtype: float32, same shape as the input;
 * - axis: one normalized integer axis, with NumPy-compatible negative-axis handling;
 * - other dtypes, malformed axes, and unsupported signatures are rejected by the
 *   implementation and are never silently cast or delegated to a host fallback.
 *
 * The L0 API is called by the ACLNN L2 API after parameter validation and
 * contiguous-tensor normalization. The L0 implementation owns shape inference,
 * output allocation, and Ascend C kernel launch.
 */

#ifndef OP_API_INC_LEVEL0_FLIP_H_
#define OP_API_INC_LEVEL0_FLIP_H_

#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Apply NumPy-compatible flip over one axis of a float32 tensor.
 *
 * @param x [in] Input tensor. The current draft supports DT_FLOAT tensors whose
 *               rank is at least one and no greater than the ACLNN maximum
 *               supported rank. The tensor must contain a valid shape.
 * @param axis [in] Axis to reverse. Values in [-rank, rank - 1] are normalized
 *                  against the input rank. An out-of-range axis is rejected.
 * @param executor [in] Op executor used for output allocation and kernel launch.
 * @return The allocated output tensor on success, or nullptr on failure.
 *
 * @note The returned output has the same shape and dtype as the input. The L2
 *       API is responsible for copying this contiguous result into a possibly
 *       non-contiguous user output tensor.
 * @note This declaration intentionally exposes only the observed float32,
 *       single-axis API shape. Broader dtype or multi-axis support requires a
 *       separate contract extension and must not be inferred here.
 */
const aclTensor* Flip(const aclTensor* x, int64_t axis, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_FLIP_H_