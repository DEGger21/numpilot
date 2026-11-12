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
 * @file aclnn_tril.h
 * @brief ACLNN L2 API declarations for the specialized tril operator.
 *
 * Semantic source: numpy.tril(a) with the observed one-positional-argument
 * signature. This candidate admits exactly one input tensor and one output
 * tensor. The diagonal offset is fixed to the NumPy default k = 0.
 *
 * The public AsNumpy API is asnumpy.tril. The native ACLNN entry points are:
 *   - aclnnTrilGetWorkspaceSize
 *   - aclnnTril
 *
 * ACLNN uses a two-stage invocation:
 *   1. aclnnTrilGetWorkspaceSize validates parameters, creates an executor,
 *      and reports the workspace requirement.
 *   2. aclnnTril executes the registered Tril kernel on the supplied stream.
 *
 * Candidate support scope:
 *   - input dtype: float32
 *   - output dtype: float32
 *   - observed input shape: rank 2, [2048, 2048]
 *   - observed layout: C-contiguous
 *   - keyword arguments: none
 *
 * Inputs outside this bounded candidate scope must be rejected explicitly by
 * the implementation. No implicit dtype conversion or CPU fallback is part of
 * this interface. Qualification remains pending until all required build,
 * numerical, native-execution, whole-program, and activation gates pass.
 */

#ifndef ACLNN_TRIL_H_
#define ACLNN_TRIL_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute the workspace size and create the executor for aclnnTril.
 *
 * @param self [in] Input tensor. The candidate supports float32 tensors whose
 *                  runtime profile has been admitted by the validation layer.
 * @param out [in] Output tensor. The output has the same shape and dtype as
 *                 the admitted float32 input. Elements on and below the main
 *                 diagonal are copied from the input; elements above the main
 *                 diagonal are set to zero.
 * @param workspaceSize [out] Required workspace size in bytes. May be zero.
 * @param executor [out] Created ACLNN executor for the second-stage call.
 * @return aclnnStatus ACLNN_SUCCESS on success, otherwise an ACLNN error code.
 */
ACLNN_API aclnnStatus aclnnTrilGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Execute the tril operator using an executor created by
 *        aclnnTrilGetWorkspaceSize.
 *
 * The caller must provide the workspace reported by the first-stage call and
 * must use the same ACL stream on which the output tensor is intended to be
 * consumed. Synchronization and error handling are performed by the ACLNN
 * execution path; a non-success status indicates that the launch or execution
 * did not complete successfully.
 *
 * @param workspace [in] Workspace memory address. May be nullptr when the
 *                       reported workspace size is zero.
 * @param workspaceSize [in] Workspace size in bytes.
 * @param executor [in] Executor returned by aclnnTrilGetWorkspaceSize.
 * @param stream [in] ACL stream used to execute the kernel or transfer work.
 * @return aclnnStatus ACLNN_SUCCESS on success, otherwise an ACLNN error code.
 */
ACLNN_API aclnnStatus aclnnTril(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_TRIL_H_