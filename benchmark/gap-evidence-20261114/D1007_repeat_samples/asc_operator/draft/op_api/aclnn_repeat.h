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
 * @file aclnn_repeat.h
 * @brief ACLNN L2 API declarations for the Repeat operator.
 *
 * Semantic contract:
 *   result = numpy.repeat(self, repeats, axis=axis)
 *
 * The public Python signature preserves the NumPy oracle order:
 *   asnumpy.repeat(input, repeats, axis=...)
 *
 * This candidate is declared for float32 tensor input and float32 tensor
 * output. The observed qualified profile is a rank-2, C-contiguous
 * [1024, 1024] input with repeats=3 and axis=1, producing [1024, 3072].
 * Unobserved dtypes and shapes remain unresolved until numerical and device
 * qualification; callers must reject them rather than silently casting or
 * falling back to a host implementation.
 *
 * The Python wrapper normalizes a negative axis to a non-negative dimension
 * index before entering this API. axis=None and array-valued repeats are
 * outside the current bounded support scope and must be rejected explicitly by
 * the public adapter before this ACLNN call is made.
 *
 * Two-stage ACLNN usage:
 *   1. aclnnRepeatGetWorkspaceSize(...)
 *   2. aclnnRepeat(workspace, workspaceSize, executor, stream)
 */

#ifndef ACLNN_REPEAT_H_
#define ACLNN_REPEAT_H_

#include <cstdint>

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute the workspace size and build the execution plan for Repeat.
 *
 * Required preconditions:
 * - self is non-null, has rank >= 1, and has dtype DT_FLOAT.
 * - repeats is a positive int64_t scalar.
 * - axis is already normalized to the range [0, self rank).
 * - out is non-null, has dtype DT_FLOAT, has the same rank as self, and has
 *   the NumPy repeat output shape on axis.
 *
 * The implementation must return ACLNN_ERR_PARAM_NULLPTR for null required
 * pointers, ACLNN_ERR_PARAM_INVALID for unsupported dtype/format/shape or
 * invalid repeats/axis, and ACLNN_ERR_INNER_* for executor or kernel-launch
 * failures. No host-side NumPy computation is permitted.
 *
 * @param self [in] Input tensor. This candidate accepts float32 data.
 * @param repeats [in] Scalar number of repetitions for each element on axis.
 * @param axis [in] Normalized non-negative dimension along which to repeat.
 * @param out [in] Preallocated output tensor with dtype float32 and the
 *                 NumPy repeat result shape.
 * @param workspaceSize [out] Number of workspace bytes required.
 * @param executor [out] Executor describing the pending device work.
 * @return ACLNN status code.
 */
ACLNN_API aclnnStatus aclnnRepeatGetWorkspaceSize(
    const aclTensor *self,
    int64_t repeats,
    int64_t axis,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the Repeat operation represented by executor.
 *
 * The caller must pass the same workspace/executor pair returned by
 * aclnnRepeatGetWorkspaceSize and must synchronize the stream before reading
 * output results. The implementation must validate workspace and executor
 * pointers, launch the registered Repeat kernel on stream, and propagate any
 * launch or execution error through the returned aclnnStatus.
 *
 * @param workspace [in] Device workspace address, or nullptr when size is 0.
 * @param workspaceSize [in] Workspace size returned by the first stage.
 * @param executor [in] Executor returned by the first stage.
 * @param stream [in] ACL stream on which to enqueue the device work.
 * @return ACLNN status code.
 */
ACLNN_API aclnnStatus aclnnRepeat(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_REPEAT_H_