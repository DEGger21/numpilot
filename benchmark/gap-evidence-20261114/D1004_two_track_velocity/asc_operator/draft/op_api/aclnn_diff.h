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
 * @file aclnn_diff.h
 * @brief ACLNN L2 API declaration for the Diff operator.
 *
 * Semantic counterpart: numpy.diff(a, axis=axis) with n fixed to 1.
 *
 * This draft is scoped to the observed call contract:
 *   - one positional tensor input
 *   - keyword attribute axis
 *   - one tensor output
 *
 * The implementation accepts only the admitted float32/rank-2/axis-1 profile.
 * Unsupported dtypes, ranks, shapes, or axis values must be rejected by the
 * GetWorkspaceSize implementation with an ACLNN parameter error. No implicit
 * dtype casting and no CPU fallback are permitted.
 *
 * ACLNN uses a two-stage call sequence:
 *   1. aclnnDiffGetWorkspaceSize(...)
 *   2. aclnnDiff(...)
 */

#ifndef ACLNN_DIFF_H_
#define ACLNN_DIFF_H_

#include <cstdint>

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute the workspace size required by aclnnDiff.
 *
 * Validates the input tensor, output tensor shape/dtype, and normalized axis.
 * The only currently admitted semantic profile is float32 rank-2 input with
 * axis=1, producing a float32 rank-2 output whose last dimension is one less
 * than the input. Other profiles must be rejected explicitly.
 *
 * @param[in]  self          Input tensor. Supported profile: float32, rank 2.
 * @param[in]  axis          Reduction/difference axis. Supported value: 1.
 * @param[in]  out           Output tensor. Must be float32 with shape
 *                           [self.shape[0], self.shape[1] - 1] for axis=1.
 * @param[out] workspaceSize Returned workspace size in bytes.
 * @param[out] executor      Returned ACLNN op executor.
 * @return ACLNN_SUCCESS on success, otherwise an ACLNN error code.
 */
ACLNN_API aclnnStatus aclnnDiffGetWorkspaceSize(
    const aclTensor *self,
    int64_t axis,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute aclnnDiff on the given ACL stream.
 *
 * The executor must have been produced by a successful call to
 * aclnnDiffGetWorkspaceSize with the same input, axis, and output tensors.
 *
 * @param[in] workspace     Workspace address. May be nullptr when workspaceSize is 0.
 * @param[in] workspaceSize Workspace size returned by aclnnDiffGetWorkspaceSize.
 * @param[in] executor      Executor returned by aclnnDiffGetWorkspaceSize.
 * @param[in] stream        ACL stream used for execution.
 * @return ACLNN_SUCCESS on success, otherwise an ACLNN error code.
 */
ACLNN_API aclnnStatus aclnnDiff(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_DIFF_H_