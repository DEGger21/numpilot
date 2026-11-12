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
 * @file aclnn_pad.h
 * @brief ACLNN L2 API declarations for the Pad operator.
 *
 * This header exposes only the two-stage ACLNN interface:
 * - aclnnPadGetWorkspaceSize: validates arguments and creates the executor.
 * - aclnnPad: executes the previously prepared operation on an ACL stream.
 *
 * The public AsNumpy API is asnumpy.pad. The current bounded implementation
 * admits float32 C-contiguous tensors and the observed call contract
 * numpy.pad(input, pad_width, mode="constant"). Other dtypes, modes, ranks, or
 * layouts are rejected by the L2 implementation rather than silently cast or
 * routed to a host fallback.
 *
 * The @p pads array is a flattened row-major representation of NumPy pad_width:
 *   [before_0, after_0, before_1, after_1, ...]
 * and must contain exactly 2 * rank entries for @p self. For the observed
 * callsite numpy.pad(a, ((1, 1), (2, 2)), mode="constant"), rank is 2 and pads
 * is [1, 1, 2, 2].
 *
 * The operation has one input tensor and one output tensor. The output shape is
 * inferred as:
 *   out_shape[i] = self_shape[i] + pads[2 * i] + pads[2 * i + 1]
 * using the NumPy ordering of dimensions.
 *
 * @note This file is an unverified candidate artifact. Numerical qualification
 * must compare against the NumPy oracle on the target Ascend910B3 device with
 * CANN 9.2.0-beta.2 before this interface is described as verified.
 */

#ifndef ACLNN_PAD_H_
#define ACLNN_PAD_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate workspace requirements and prepare a Pad execution.
 *
 * Supported contract for this draft stage:
 * - @p self dtype: float32.
 * - @p out dtype: float32.
 * - @p self and @p out layout: contiguous (the L2 implementation will reject
 *   unsupported layouts or use the established contiguous conversion path).
 * - @p mode: "constant" only. The constant value is zero, matching NumPy's
 *   default constant_values=0 for the observed callsite.
 * - @p pads: flattened before/after pairs in row-major dimension order, with
 *   exactly 2 * rank values.
 *
 * Argument validation, unsupported-dtype rejection, unsupported-mode rejection,
 * shape inference, and inference of the output shape are the responsibility of
 * the L2 implementation. This declaration deliberately does not accept a
 * constant_values argument because the admitted Python contract has only two
 * positional arguments plus the mode keyword.
 *
 * The returned executor owns all intermediate tensors and attributes needed by
 * aclnnPad. The caller must not destroy it before the execute phase completes.
 *
 * @param[in] self Input tensor. Must be non-null and have supported dtype and
 *                 layout.
 * @param[in] pads Flattened NumPy-style pad widths, size 2 * self rank. Must be
 *                 non-null and contain non-negative entries.
 * @param[in] mode Padding mode string. The current bounded implementation
 *                 accepts "constant" and rejects other values explicitly. Must
 *                 be non-null.
 * @param[in] out Preallocated output tensor. Must be non-null, have supported
 *                dtype and layout, and have the inferred padded shape.
 * @param[out] workspaceSize Receives the workspace size in bytes required by
 *                           the execute phase. Must be non-null.
 * @param[out] executor Receives the prepared executor. Must be non-null.
 *
 * @return ACLNN_SUCCESS on successful preparation; otherwise an ACLNN error
 *         code. Parameters are rejected with ACLNN_ERR_PARAM_INVALID or the
 *         applicable null/executor error code. Callers must check the return
 *         value before invoking aclnnPad.
 */
ACLNN_API aclnnStatus aclnnPadGetWorkspaceSize(
    const aclTensor* self,
    const aclIntArray* pads,
    const char* mode,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief Execute a prepared Pad operation.
 *
 * This function enqueues the operation on @p stream. It does not implicitly
 * synchronize the stream. The caller is responsible for checking the returned
 * status and, when a synchronous result is required, for calling the
 * appropriate ACL synchronization primitive after this function returns.
 *
 * If @p workspaceSize is zero, @p workspace may be null. Otherwise @p workspace
 * must reference at least @p workspaceSize bytes. @p executor must be the
 * executor produced by aclnnPadGetWorkspaceSize for the same arguments and must
 * remain valid until execution has completed.
 *
 * @param[in] workspace Workspace memory address, or null when the prepared
 *                      workspace size is zero.
 * @param[in] workspaceSize Workspace size returned by
 *                          aclnnPadGetWorkspaceSize.
 * @param[in] executor Prepared executor returned by
 *                     aclnnPadGetWorkspaceSize.
 * @param[in] stream ACL stream on which to enqueue execution.
 *
 * @return ACLNN_SUCCESS when execution is successfully enqueued; otherwise an
 *         ACLNN error code. A returned success indicates enqueue success, not
 *         completion of device execution.
 */
ACLNN_API aclnnStatus aclnnPad(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_PAD_H_