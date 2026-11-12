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
 * @file aclnn_flip.h
 * @brief ACLNN L2 API declaration for the Flip operator.
 *
 * Semantic target: numpy.flip(a, axis=...)
 * Public AsNumpy API: asnumpy.flip(a, axis=...)
 * C++ OpType: Flip
 * File/API stem: flip
 *
 * Name mapping:
 * - NumPy positional argument `a`  -> ACLNN input tensor `x`
 * - NumPy keyword argument `axis`  -> ACLNN scalar attribute `axis`
 * - NumPy return value             -> ACLNN output tensor `out`
 *
 * ACLNN uses the standard two-stage interface:
 * 1. aclnnFlipGetWorkspaceSize() validates parameters and builds the executor.
 * 2. aclnnFlip() enqueues execution on the caller-provided ACL stream.
 *
 * Candidate support scope for this generation:
 * - Input dtype: float32 (ACLNN DT_FLOAT) only. No implicit casts are allowed.
 * - Output dtype must be exactly the input dtype; for this candidate, float32.
 * - Axis: one signed integer axis. Negative axes are normalized by the host
 *   implementation. axis=None and multiple-axis tuples are not representable in
 *   this ACLNN signature and must be rejected by the Python wrapper before the
 *   ACLNN call rather than silently converted.
 * - The observed candidate profile is a rank-2, C-contiguous float32 tensor with
 *   shape [2048, 2048] and axis=1. Additional float32 shape/rank profiles are not
 *   qualified by this declaration; the L2 implementation must validate them at
 *   runtime against its declared bounded support and return ACLNN_ERR_PARAM_INVALID
 *   for unsupported cases.
 *
 * Error handling:
 * - Null tensor, workspaceSize, or executor pointers produce ACLNN_ERR_PARAM_NULLPTR.
 * - Unsupported dtype, unsupported storage format, incompatible output dtype/shape,
 *   out-of-range axis, or unsupported rank/shape produce ACLNN_ERR_PARAM_INVALID.
 * - Internal executor/kernel allocation failures produce the corresponding
 *   ACLNN_ERR_INNER_* status.
 *
 * Synchronization:
 * - aclnnFlip() is asynchronous with respect to the host and runs on the supplied
 *   aclrtStream. The caller owns stream synchronization and must check the returned
 *   aclnnStatus before consuming the output tensor.
 *
 * Workspace lifecycle:
 * - workspaceSize returned by aclnnFlipGetWorkspaceSize() is exactly the size the
 *   caller must allocate for the second call. A zero value means no workspace is
 *   required. The executor is consumed by the second call and must not be reused.
 *
 * This file is an unverified generation-stage declaration. It does not assert
 * numerical or device qualification; those gates remain pending.
 */

#ifndef ACLNN_FLIP_H_
#define ACLNN_FLIP_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute the workspace size and create the executor for aclnnFlip.
 *
 * @param[in]  x              Input tensor. Required. Must be DT_FLOAT for the
 *                            current candidate scope and must have a supported
 *                            rank/shape/format as validated by the L2 layer.
 * @param[in]  axis           Single signed flip axis. Negative values are
 *                            normalized against the input rank by the host
 *                            implementation. The caller must supply exactly one
 *                            integer axis; axis=None or tuple axes are rejected
 *                            by the Python wrapper before this call.
 * @param[in]  out            Output tensor. Required. Must be DT_FLOAT, have the
 *                            same shape as `x`, and be a supported non-private
 *                            format. The L2 layer handles non-contiguous output
 *                            through ViewCopy when supported.
 * @param[out] workspaceSize  Returns the exact workspace byte size required by
 *                            the second-stage aclnnFlip call. Must not be null.
 * @param[out] executor       Returns the prepared executor. Must not be null.
 *
 * @return ACLNN_SUCCESS on success; otherwise an ACLNN error status. On failure,
 *         workspaceSize and executor are not usable for execution.
 *
 * @note The output is not filled by this call. Execution occurs only when
 *       aclnnFlip() is invoked with the returned executor and a valid stream.
 * @note For empty input tensors, the executor may represent a no-op; the caller
 *       must still allocate according to the returned workspaceSize and invoke
 *       aclnnFlip() to preserve the ACLNN two-stage lifecycle.
 */
ACLNN_API aclnnStatus aclnnFlipGetWorkspaceSize(
    const aclTensor *x,
    int64_t axis,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the Flip computation previously prepared by
 *        aclnnFlipGetWorkspaceSize().
 *
 * @param[in] workspace      Workspace memory of at least workspaceSize bytes.
 *                           May be null only when workspaceSize is zero.
 * @param[in] workspaceSize  Workspace size returned by the first-stage call.
 * @param[in] executor       Executor returned by the first-stage call. Ownership
 *                           is consumed by this call.
 * @param[in] stream         ACL stream on which the kernel is enqueued.
 *
 * @return ACLNN_SUCCESS when execution is successfully enqueued; otherwise an
 *         ACLNN error status. The caller must synchronize the stream and check
 *         this status before reading `out`.
 *
 * @note The output tensor must remain valid and unchanged in shape/dtype until
 *       the enqueued work completes.
 * @note This function performs no host fallback and no implicit dtype conversion.
 */
ACLNN_API aclnnStatus aclnnFlip(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_FLIP_H_