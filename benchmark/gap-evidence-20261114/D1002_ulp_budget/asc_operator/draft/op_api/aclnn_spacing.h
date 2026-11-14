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
 * @file aclnn_spacing.h
 * @brief ACLNN L2 API declarations for the unary Spacing operator.
 *
 * Semantic contract:
 *   out = spacing(x)
 *
 * This ACLNN entry point follows the standard two-stage design:
 *   1. aclnnSpacingGetWorkspaceSize creates an executor and validates parameters.
 *   2. aclnnSpacing executes the registered Spacing operator on the supplied stream.
 *
 * Supported profile for this candidate:
 *   - One positional input tensor.
 *   - No keyword arguments.
 *   - float32 input and float32 output with identical shapes.
 *   - C-contiguous tensors; non-contiguous inputs are made contiguous by L2.
 *
 * Unsupported input dtypes and signature variants are rejected explicitly by
 * the implementation. The output retains IEEE-754 special values, including
 * NaNs, infinities, subnormals, and signed zero, as defined by the NumPy
 * np.spacing oracle.
 */

#ifndef ACLNN_SPACING_H_
#define ACLNN_SPACING_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute the workspace size and build the execution plan for Spacing.
 *
 * The input and output tensors must be non-null. For this candidate, both
 * tensors must have dtype ACL_FLOAT and the same shape. The required output
 * shape and dtype are also checked by shape and dtype inference in the host
 * implementation and by the L0 ACLNN wrapper.
 *
 * An empty input tensor is a valid shape case. In that case no kernel work is
 * required and zero workspace is reported.
 *
 * @param x              [in]  Input float32 tensor.
 * @param out            [in]  Output float32 tensor with the same shape as x.
 * @param workspaceSize  [out] Required workspace size in bytes.
 * @param executor       [out] Executor containing the staged operation.
 * @return ACLNN_SUCCESS on success; an ACLNN error code on invalid parameters
 *         or internal allocation failure.
 */
ACLNN_API aclnnStatus aclnnSpacingGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the staged Spacing operation.
 *
 * The caller must invoke aclnnSpacingGetWorkspaceSize first and pass its
 * executor and workspace size unchanged. This function performs the final
 * parameter check and enqueues the kernel on the provided ACL stream. Work is
 * completed asynchronously according to ACL stream semantics; callers that
 * need host-visible completion must synchronize the stream.
 *
 * @param workspace      [in] Workspace pointer, or nullptr when size is zero.
 * @param workspaceSize  [in] Workspace size returned by GetWorkspaceSize.
 * @param executor       [in] Executor returned by GetWorkspaceSize.
 * @param stream         [in] ACL stream used to execute the operator.
 * @return ACLNN_SUCCESS on accepted execution; an ACLNN error code otherwise.
 */
ACLNN_API aclnnStatus aclnnSpacing(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SPACING_H_