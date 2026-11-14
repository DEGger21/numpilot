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
 * @file aclnn_roll.h
 * @brief ACLNN L2 API declarations for Roll.
 *
 * Public mapping:
 *   Python public API : asnumpy.roll(input, shift, axis=...)
 *   ACLNN L2 symbols  : aclnnRollGetWorkspaceSize, aclnnRoll
 *   C++ OpType        : Roll
 *
 * The public NumPy argument order is preserved:
 *   input  - tensor to rotate
 *   shift  - number of positions to rotate
 *   axis   - axis along which elements are rotated
 *   out    - destination tensor supplied by the ACLNN caller
 *
 * This unverified candidate targets the observed float32 input profile and
 * supports a scalar shift with a scalar axis. Unsupported dtypes, axis forms,
 * layouts, and shapes must be rejected explicitly by the implementation rather
 * than implicitly cast, silently reinterpreted, or executed on the host.
 *
 * ACLNN uses a two-stage call contract:
 *   1. aclnnRollGetWorkspaceSize performs validation and builds the executor.
 *   2. aclnnRoll executes the prepared executor on the supplied ACL stream.
 */

#ifndef ACLNN_ROLL_H_
#define ACLNN_ROLL_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validate parameters and compute the workspace required by aclnnRoll.
 *
 * The implementation checks that the input and output tensors are non-null,
 * have matching supported dtypes, carry a supported layout and rank, and have
 * compatible shapes. It normalizes the scalar axis against the input rank and
 * reduces the scalar shift modulo the selected axis extent. For non-contiguous
 * admitted inputs, the implementation is responsible for materializing a
 * contiguous device representation before invoking the Roll kernel.
 *
 * @param self          [in]  Input tensor (NumPy roll argument 0).
 * @param shift         [in]  Rotation amount (NumPy roll argument 1).
 * @param axis          [in]  Axis along which to rotate (NumPy roll keyword).
 * @param out           [in]  Output tensor supplied by the caller.
 * @param workspaceSize [out] Workspace size in bytes required by aclnnRoll.
 * @param executor      [out] Executor prepared for aclnnRoll.
 * @return ACLNN_SUCCESS on success; an ACLNN error code otherwise.
 */
ACLNN_API aclnnStatus aclnnRollGetWorkspaceSize(
    const aclTensor *self,
    int64_t shift,
    int64_t axis,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the Roll computation prepared by aclnnRollGetWorkspaceSize.
 *
 * The caller must pass the workspaceSize and executor returned by the
 * corresponding first-stage call. Execution is asynchronous with respect to the
 * host and is submitted to the supplied ACL stream. The caller is responsible
 * for stream synchronization before observing output data on the host, exactly
 * as required by the ACLNN two-stage API contract.
 *
 * @param workspace     [in] Workspace memory; may be null when size is zero.
 * @param workspaceSize [in] Workspace size returned by the first-stage call.
 * @param executor      [in] Executor returned by the first-stage call.
 * @param stream        [in] ACL stream used to submit the computation.
 * @return ACLNN_SUCCESS on successful submission; an ACLNN error code otherwise.
 */
ACLNN_API aclnnStatus aclnnRoll(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ROLL_H_