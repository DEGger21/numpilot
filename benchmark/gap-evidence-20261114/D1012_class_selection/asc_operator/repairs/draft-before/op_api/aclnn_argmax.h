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
 * @file aclnn_argmax.h
 * @brief ACLNN L2 API declarations for Argmax.
 *
 * Public NumPy mapping:
 *   numpy.argmax(input, axis=axis) -> asnumpy.argmax(input, axis=axis)
 *
 * ACLNN two-stage mapping:
 *   aclnnArgmaxGetWorkspaceSize -> validates parameters and builds the executor
 *   aclnnArgmax                -> executes the registered Argmax kernel
 *
 * Declared bounded support scope for this draft:
 *   - self: float32 tensor
 *   - axis: int64 scalar attribute, normalized against self rank
 *   - out: int64 tensor with the selected axis removed
 *   - signature: one positional tensor plus the axis keyword, matching the
 *     NumPy oracle parameter order used by the AsNumpy adapter
 *
 * Unsupported input dtypes and invalid axes must be rejected by the L2
 * implementation. This declaration does not qualify any dtype or shape profile
 * before numerical and device validation.
 */

#ifndef ACLNN_ARGMAX_H_
#define ACLNN_ARGMAX_H_

#include <cstdint>

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute the workspace size and build the executor for aclnnArgmax.
 *
 * @param self          [in]  Input tensor. This draft targets float32 tensors.
 * @param axis          [in]  Reduction axis. It is normalized by the L2
 *                            implementation using the rank of self.
 * @param out           [in]  Output tensor. This draft requires int64 output;
 *                            its shape must equal the input shape with axis
 *                            removed.
 * @param workspaceSize [out] Required workspace size in bytes. May be zero.
 * @param executor      [out] Constructed ACLNN executor.
 * @return ACLNN_SUCCESS on success, otherwise an ACLNN error code.
 *
 * The implementation must reject null tensors, unsupported self/output dtypes,
 * private formats, invalid axes, and shape mismatches. It must not silently cast
 * other input dtypes to float32 or fall back to a host implementation.
 */
ACLNN_API aclnnStatus aclnnArgmaxGetWorkspaceSize(
    const aclTensor *self,
    int64_t axis,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the prepared aclnnArgmax computation.
 *
 * @param workspace     [in] Workspace address returned by the first stage.
 * @param workspaceSize [in] Workspace size returned by the first stage.
 * @param executor      [in] Executor returned by the first stage.
 * @param stream        [in] ACL stream used for execution.
 * @return ACLNN_SUCCESS on success, otherwise an ACLNN error code.
 *
 * The caller owns synchronization of stream and workspace lifetime. Errors from
 * executor execution must be propagated without reporting success.
 */
ACLNN_API aclnnStatus aclnnArgmax(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ARGMAX_H_