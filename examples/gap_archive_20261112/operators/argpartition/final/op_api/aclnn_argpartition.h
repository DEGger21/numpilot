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
 * @file aclnn_argpartition.h
 * @brief ACLNN L2 API declarations for Argpartition.
 *
 * This interface preserves NumPy argpartition call semantics:
 *   argpartition(a, kth, axis=...)
 *
 * API mapping:
 *   asnumpy.argpartition
 *     -> Python wrapper (argpartition)
 *     -> native ACLNN L2 API (aclnnArgpartitionGetWorkspaceSize / aclnnArgpartition)
 *     -> L0 API (Argpartition)
 *     -> kernel entry (argpartition)
 *
 * The generated candidate admits the observed float32 profile only:
 *   input dtype float32, rank 2, shape [32, 16], C-contiguous
 *   kth = 3, axis = 1
 *   output dtype int64, shape [32, 16]
 *
 * Unsupported dtypes, shapes, kth values, and axes must be rejected explicitly
 * by the implementation. No implicit dtype cast or CPU fallback is permitted.
 */

#ifndef ACLNN_ARGPARTITION_H_
#define ACLNN_ARGPARTITION_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the workspace size and create the executor for aclnnArgpartition.
 *
 * The output tensor is a required argument and must have dtype int64 with the
 * same shape as the input tensor for the admitted profile.
 *
 * @param self          [in]  Input tensor (float32 for this candidate).
 * @param kth           [in]  Zero-based partition index. For the observed
 *                            profile, kth = 3.
 * @param axis          [in]  Partition axis. Negative values are normalized by
 *                            the implementation; for the observed profile,
 *                            axis = 1.
 * @param out           [in]  Output tensor (int64, shape matching self).
 * @param workspaceSize [out] Required workspace size in bytes.
 * @param executor      [out] Created operation executor.
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnArgpartitionGetWorkspaceSize(
    const aclTensor *self,
    int64_t kth,
    int64_t axis,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the Argpartition operation.
 *
 * This is the second stage of the ACLNN two-stage call. The implementation
 * performs explicit ACL error checking and stream synchronization as required
 * by the operator project.
 *
 * @param workspace     [in] Workspace memory address.
 * @param workspaceSize [in] Workspace size in bytes.
 * @param executor      [in] Executor from aclnnArgpartitionGetWorkspaceSize.
 * @param stream        [in] ACL stream used for execution.
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnArgpartition(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ARGPARTITION_H_