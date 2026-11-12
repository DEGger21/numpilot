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
 * @file aclnn_ediff1d.h
 * @brief ACLNN L2 API declarations for NumPy-compatible ediff1d.
 *
 * This header declares the public two-stage ACLNN interface for the
 * accuracy-first Ediff1d registry-invoke operator:
 *
 *   NumPy:  numpy.ediff1d(a)
 *   AsNumpy: asnumpy.ediff1d(a)
 *   ACLNN:  aclnnEdiff1dGetWorkspaceSize(x, out, workspaceSize, executor)
 *           aclnnEdiff1d(workspace, workspaceSize, executor, stream)
 *
 * Semantics follow numpy.ediff1d for the admitted profile:
 *   - The float32 input is flattened in C order.
 *   - The output is a one-dimensional float32 tensor containing adjacent
 *     differences of the flattened input.
 *   - For the observed 2048x2048 input, the output shape is [4194303].
 *
 * Supported scope for this unverified candidate:
 *   - Input dtype: float32.
 *   - Output dtype: float32.
 *   - Input layout: ND; non-contiguous inputs are handled by the L2
 *     implementation with l0op::Contiguous before kernel invocation.
 *   - Signature: exactly one positional argument, no keyword arguments.
 *     Optional NumPy arguments such as to_begin and to_end are intentionally
 *     not exposed by this header and are rejected by the Python/adapter layer.
 *
 * Unsupported input dtypes must be rejected explicitly by the implementation;
 * they must not be silently cast to float32. Unknown shapes and layouts remain
 * outside the qualified scope until deterministic device tests establish them.
 *
 * ACLNN interfaces use the standard two-stage design:
 *   - GetWorkspaceSize: validates parameters, prepares the executor, applies
 *     contiguous/view-copy handling, and reports workspace requirements.
 *   - aclnnEdiff1d: launches the prepared executor on the supplied ACL stream.
 *
 * File naming convention:
 *   - L2 API: aclnn_{op}.h/cpp
 *   - L0 API: {op}.h/cpp
 */

#ifndef ACLNN_EDIFF1D_H_
#define ACLNN_EDIFF1D_H_

#include <cstdint>

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate the workspace size required by aclnnEdiff1d.
 *
 * The input corresponds to the single positional NumPy argument a. The output
 * tensor must be a one-dimensional float32 tensor whose length matches the
 * flattened adjacent-difference result inferred by the host implementation.
 *
 * For the observed profile:
 *   x:   float32, shape [2048, 2048], C-contiguous
 *   out: float32, shape [4194303], C-contiguous
 *
 * Parameter and dtype validation is performed in the implementation. Inputs
 * whose dtype is not float32 must be rejected with ACLNN_ERR_PARAM_INVALID
 * rather than converted implicitly.
 *
 * @param x             [in]  Float32 input tensor, NumPy parameter a.
 * @param out           [in]  Float32 one-dimensional output tensor.
 * @param workspaceSize [out] Required workspace size in bytes.
 * @param executor      [out] Prepared ACLNN executor for the second stage.
 * @return aclnnStatus ACLNN_SUCCESS on success, otherwise an ACLNN error code.
 */
ACLNN_API aclnnStatus aclnnEdiff1dGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the prepared Ediff1d computation.
 *
 * The caller must pass the workspace and executor produced by
 * aclnnEdiff1dGetWorkspaceSize. The implementation dispatches the Ascend C
 * kernel, propagates launch and execution errors, and synchronizes according
 * to the ACLNN execution contract before returning a status.
 *
 * @param workspace     [in] Workspace memory address.
 * @param workspaceSize [in] Workspace size in bytes.
 * @param executor      [in] Executor returned by the first-stage call.
 * @param stream        [in] ACL stream used for execution.
 * @return aclnnStatus ACLNN_SUCCESS on success, otherwise an ACLNN error code.
 */
ACLNN_API aclnnStatus aclnnEdiff1d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_EDIFF1D_H_