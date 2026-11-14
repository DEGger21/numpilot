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
 * @file aclnn_nextafter.cpp
 * @brief ACLNN L2 API implementation for asnumpy.nextafter / numpy.nextafter.
 *
 * Public mapping for this candidate:
 * - Python: asnumpy.nextafter(x1, x2)
 * - pybind/native adapter: nextafter(x1, x2)
 * - ACLNN L2: aclnnNextafterGetWorkspaceSize / aclnnNextafter
 * - ACLNN L0: l0op::Nextafter
 *
 * Supported draft scope:
 * - exactly two tensor inputs and one tensor output;
 * - float32 inputs and float32 output only;
 * - identical input/output view shapes, no broadcasting;
 * - non-private storage formats with matching formats.
 *
 * Other dtypes and broadcast shapes are rejected explicitly. No implicit
 * cast, CPU fallback, or host computation is performed here.
 */

#include "aclnn_nextafter.h"
#include "nextafter.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK(x1->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter only supports float32 input; x1 dtype=%d.",
                     static_cast<int>(x1->GetDataType())),
             return false);
    OP_CHECK(x2->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter only supports float32 input; x2 dtype=%d.",
                     static_cast<int>(x2->GetDataType())),
             return false);
    OP_CHECK(out->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter only supports float32 output; out dtype=%d.",
                     static_cast<int>(out->GetDataType())),
             return false);
    OP_CHECK(x1->GetDataType() == x2->GetDataType() &&
             x1->GetDataType() == out->GetDataType(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter input/output dtypes must match: x1=%d, x2=%d, out=%d.",
                     static_cast<int>(x1->GetDataType()),
                     static_cast<int>(x2->GetDataType()),
                     static_cast<int>(out->GetDataType())),
             return false);
    OP_CHECK(IsDtypeSupported(x1->GetDataType()) &&
             IsDtypeSupported(x2->GetDataType()) &&
             IsDtypeSupported(out->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter dtype not supported. Supported dtype: FLOAT (float32)."),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    const auto format1 = x1->GetStorageFormat();
    const auto format2 = x2->GetStorageFormat();
    const auto formatOut = out->GetStorageFormat();

    OP_CHECK(format1 == format2 && format1 == formatOut,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter requires matching storage formats: x1=%d, x2=%d, out=%d.",
                     static_cast<int>(format1),
                     static_cast<int>(format2),
                     static_cast<int>(formatOut)),
             return false);

    OP_CHECK(!(IsPrivateFormat(format1) || IsPrivateFormat(format2) || IsPrivateFormat(formatOut)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter does not support private storage formats: x1=%d, x2=%d, out=%d.",
                     static_cast<int>(format1),
                     static_cast<int>(format2),
                     static_cast<int>(formatOut)),
             return false);
    return true;
}

static bool SameShape(const op::Shape& a, const op::Shape& b)
{
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x1, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(x2, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    const op::Shape& shape1 = x1->GetViewShape();
    const op::Shape& shape2 = x2->GetViewShape();
    const op::Shape& shapeOut = out->GetViewShape();

    OP_CHECK(SameShape(shape1, shape2) && SameShape(shape1, shapeOut),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Nextafter requires identical input/output shapes; broadcasting is not supported. "
                     "x1 dim=%zu, x2 dim=%zu, out dim=%zu.",
                     shape1.GetDimNum(),
                     shape2.GetDimNum(),
                     shapeOut.GetDimNum()),
             return false);
    return true;
}

static bool HasEmptyTensor(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    return x1->IsEmpty() || x2->IsEmpty() || out->IsEmpty();
}

static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    if (!CheckNotNull(x1, x2, out)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(x1, x2, out)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(x1, x2, out)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(x1, x2, out)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

/**
 * @brief First-stage ACLNN API: validate parameters and build the executor.
 *
 * The implementation follows the standard two-stage ACLNN flow:
 * CREATE_EXECUTOR -> CheckParams -> empty-tensor fast path -> Contiguous
 * -> l0op::Nextafter -> ViewCopy -> GetWorkspaceSize.
 */
extern "C" aclnnStatus aclnnNextafterGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNextafter, DFX_IN(x1, x2), DFX_OUT(out));

    OP_CHECK(workspaceSize != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "aclnnNextafterGetWorkspaceSize workspaceSize is null."),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "aclnnNextafterGetWorkspaceSize executor is null."),
             return ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = 0;
    *executor = nullptr;

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclnnStatus checkRet = CheckParams(x1, x2, out);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    if (HasEmptyTensor(x1, x2, out)) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::Nextafter(x1Contiguous, x2Contiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief Second-stage ACLNN API: launch the kernel through the executor.
 *
 * The ACLNN execute phase remains asynchronous by contract. Synchronization is
 * performed by the caller/adapter after this function returns; this layer does
 * not insert a host fallback or a hidden device-side computation.
 */
extern "C" aclnnStatus aclnnNextafter(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNextafter);

    OP_CHECK(executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "aclnnNextafter executor is null."),
             return ACLNN_ERR_PARAM_NULLPTR);

    const aclnnStatus ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "aclnnNextafter execution failed with status %d.",
                static_cast<int>(ret));
    }
    return ret;
}