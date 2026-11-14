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
 * @file aclnn_nextafter.h
 * @brief ACLNN L2 API declarations for the Nextafter operator.
 *
 * Semantics (NumPy oracle, np.nextafter(x1, x2)):
 *   out[i] = the representable float32 value adjacent to x1[i] in the direction
 *            of x2[i]. Signed zero, infinity and NaN behaviour follow the
 *            IEEE-754 / NumPy contract:
 *              - nextafter(a, b) with a == b (including +0.0 vs +0.0 and
 *                -0.0 vs -0.0) returns b unchanged;
 *              - nextafter(a, b) with a != b but b == a in value (e.g. +0.0 and
 *                -0.0) returns b with b's sign;
 *              - nextafter(a, NaN) and nextafter(NaN, b) return NaN;
 *              - nextafter(+-inf, b) steps to the largest finite magnitude when
 *                b is finite, and returns b for opposite-directed infinities;
 *              - stepping across zero preserves the sign of zero as NumPy does.
 *
 * Two-stage ACLNN design:
 *   - aclnnNextafterGetWorkspaceSize: validates parameters, materialises
 *     contiguous inputs, runs shape/dtype inference and returns the required
 *     workspace size plus an executor.
 *   - aclnnNextafter: executes the previously prepared plan on the given stream.
 *
 * Bounded support scope of this candidate (unverified draft):
 *   - input dtype  : float32 only, for both operands (DT_FLOAT);
 *   - output dtype : float32, identical to the input dtype;
 *   - shapes       : both operands and the output must have the same shape,
 *                    rank <= 8, non-private (ND) format;
 *   - layout       : any stride is accepted at the API boundary through
 *                    contiguous conversion, but only the C-contiguous observed
 *                    profile has been measured;
 *   - anything else (other dtypes, mismatched shapes, private formats,
 *     unsupported keywords) is rejected explicitly with ACLNN_ERR_PARAM_INVALID.
 *     No implicit dtype cast and no host/CPU fallback is performed.
 *
 * Symbol mapping:
 *   asnumpy.nextafter            (public Python API)
 *     -> asnumpy/_nextafter pybind wrapper
 *        -> aclnnNextafterGetWorkspaceSize / aclnnNextafter (this header, L2)
 *           -> l0op::Nextafter (L0 API, op_api/nextafter.h)
 *              -> Nextafter kernel entry (op_kernel/)
 */

#ifndef ACLNN_NEXTAFTER_H_
#define ACLNN_NEXTAFTER_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the workspace size and prepares the execution plan for
 *        aclnnNextafter.
 *
 * Performs, in order:
 *   1. executor creation;
 *   2. parameter checks (non-null, dtype float32, matching output dtype,
 *      ND format, equal shapes with rank <= 8);
 *   3. empty-tensor fast path (returns workspaceSize = 0);
 *   4. contiguous conversion of both inputs;
 *   5. L0 shape inference and output allocation via l0op::Nextafter;
 *   6. ViewCopy back to the user output tensor (supports user strides).
 *
 * @param x1 [in]  first operand (the value to start from), float32 aclTensor
 * @param x2 [in]  second operand (the direction target), float32 aclTensor
 * @param out [in] output tensor, float32, same shape as x1 and x2
 * @param workspaceSize [out] required workspace size in bytes
 * @param executor [out] returned op executor, owned by the caller on success
 * @return aclnnStatus ACLNN_SUCCESS on success, an ACLNN error code otherwise
 */
ACLNN_API aclnnStatus aclnnNextafterGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Executes the prepared Nextafter computation.
 *
 * The call is asynchronous with respect to the host; the work is queued on
 * @p stream and completion is observed through that stream. The caller must
 * pass the same executor that was produced by
 * aclnnNextafterGetWorkspaceSize and must keep the input/output device memory
 * alive until the stream has finished. A null executor, an insufficient
 * workspace or a stream submission failure is reported as an ACLNN error code
 * instead of being silently ignored.
 *
 * @param workspace [in] workspace device pointer (may be nullptr when
 *                       workspaceSize is 0)
 * @param workspaceSize [in] workspace size returned by the first stage
 * @param executor [in] executor produced by the first stage
 * @param stream [in] ACL stream used to launch the kernel
 * @return aclnnStatus ACLNN_SUCCESS on success, an ACLNN error code otherwise
 */
ACLNN_API aclnnStatus aclnnNextafter(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_NEXTAFTER_H_