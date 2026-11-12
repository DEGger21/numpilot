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
 * @file aclnn_std.cpp
 * @brief ACLNN L2 API 实现 - 标准差归约算子（语义源 numpy.std，公开 API asnumpy.std）
 *
 * 名称映射（与 op_host / op_kernel / pybind / adapter 保持一致）：
 *   public API      : asnumpy.std(a, axis=...)
 *   python wrapper  : asnumpy.std -> _std.std(a, axis)
 *   pybind symbol   : asnumpy._C.std
 *   native impl     : l0op::Std  (op_api/std.cpp, 内部由 AddExample 同构流程调度 kernel `Std`)
 *   ACLNN 两段式    : aclnnNumpyStdGetWorkspaceSize / aclnnNumpyStd  (本文件)
 *   C++ OpType      : NumpyStd
 *
 * 支持范围（未验证候选，仅覆盖已观测 profile）：
 *   - 输入 dtype: float32（DT_FLOAT），其他 dtype 显式拒绝，不做隐式 cast；
 *   - 输出 dtype: float32，单 tensor 返回（keepdims=False，axis 维被消除）；
 *   - 签名: 1 个位置参数 + 关键字参数 axis（单一整数轴，必须落在 [-rank, rank-1]）；
 *   - 空 tensor、0-d tensor、私有 format、rank > 8、axis 越界、输出 shape 不匹配 均显式拒绝。
 *
 * ACLNN 接口采用两段式设计：
 *   1. aclnnNumpyStdGetWorkspaceSize - 参数检查、Contiguous/ViewCopy 处理、创建执行器、返回 workspace 大小
 *   2. aclnnNumpyStd                 - 在同一 stream 上执行，返回状态码由调用方显式检查并同步
 */

#include "aclnn_std.h"
#include "std.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

// 本候选仅支持 float32：其他 dtype 必须显式拒绝，禁止静默 cast 或回退 CPU。
static const std::initializer_list<op::DataType> STD_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, STD_AICORE_DTYPE_SUPPORT_LIST);
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // NumPy: np.std(float32) -> float32；输出 dtype 必须与输入一致，不做提升。
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);

    OP_CHECK(IsDtypeSupported(self->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Std: unsupported input dtype=%d, only FLOAT (float32) is admitted; "
                     "no implicit cast and no host fallback.",
                     static_cast<int>(self->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    auto formatSelf = self->GetStorageFormat();
    auto formatOut = out->GetStorageFormat();

    OP_CHECK(!(IsPrivateFormat(formatSelf) || IsPrivateFormat(formatOut)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Std: private format not supported: self=%d, out=%d",
                     static_cast<int>(formatSelf), static_cast<int>(formatOut)),
             return false);
    return true;
}

static bool CheckRank(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static bool CheckNotEmpty(const aclTensor* self)
{
    // 空 tensor 归约语义（零元素标准差）不在本候选声明范围内，显式拒绝而不静默跳过。
    OP_CHECK(!self->IsEmpty(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Std: empty input tensor is outside the supported scope of this candidate."),
             return false);
    return true;
}

static bool NormalizeAxis(int64_t axis, int64_t rank, int64_t& normalizedAxis)
{
    OP_CHECK(rank > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Std: 0-d input tensor is not supported (axis=%ld).",
                     static_cast<long>(axis)),
             return false);

    OP_CHECK(axis >= -rank && axis < rank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Std: axis=%ld out of range for input rank %ld; axis must be in [-%ld, %ld].",
                     static_cast<long>(axis), static_cast<long>(rank),
                     static_cast<long>(rank), static_cast<long>(rank - 1)),
             return false);

    normalizedAxis = (axis < 0) ? (axis + rank) : axis;
    return true;
}

static bool CheckAxis(const aclTensor* self, int64_t axis)
{
    int64_t normalizedAxis = 0;
    const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    return NormalizeAxis(axis, rank, normalizedAxis);
}

static bool CheckOutputShape(const aclTensor* self, int64_t axis, const aclTensor* out)
{
    const auto& selfShape = self->GetViewShape();
    const auto& outShape = out->GetViewShape();
    const int64_t selfRank = static_cast<int64_t>(selfShape.GetDimNum());

    int64_t normalizedAxis = 0;
    OP_CHECK(NormalizeAxis(axis, selfRank, normalizedAxis),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Std: normalize axis failed."),
             return false);

    // keepdims=False：输出 rank 必须比输入少 1（与本候选签名一致）。
    OP_CHECK(static_cast<int64_t>(outShape.GetDimNum()) + 1 == selfRank,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Std: out rank=%zu must equal in rank-1 (in rank=%ld), keepdims=False is fixed.",
                     outShape.GetDimNum(), static_cast<long>(selfRank)),
             return false);

    int64_t outIdx = 0;
    for (int64_t i = 0; i < selfRank; ++i) {
        if (i == normalizedAxis) {
            continue;
        }
        OP_CHECK(outShape.GetDim(static_cast<size_t>(outIdx)) == selfShape.GetDim(static_cast<size_t>(i)),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "Std: out shape mismatch at out dim %ld: got %ld, expected %ld.",
                         static_cast<long>(outIdx),
                         static_cast<long>(outShape.GetDim(static_cast<size_t>(outIdx))),
                         static_cast<long>(selfShape.GetDim(static_cast<size_t>(i)))),
                 return false);
        ++outIdx;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, int64_t axis, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR, "Std: CheckNotNull failed.");
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "Std: CheckDtypeValid failed: self_dtype=%d, out_dtype=%d",
               static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID,
               "Std: CheckFormat failed: self_format=%d, out_format=%d",
               static_cast<int>(self->GetStorageFormat()), static_cast<int>(out->GetStorageFormat()));
    CHECK_COND(CheckRank(self, out), ACLNN_ERR_PARAM_INVALID,
               "Std: CheckRank failed: self_dim=%zu, out_dim=%zu",
               self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());
    CHECK_COND(CheckNotEmpty(self), ACLNN_ERR_PARAM_INVALID,
               "Std: CheckNotEmpty failed: self numel=0");
    CHECK_COND(CheckAxis(self, axis), ACLNN_ERR_PARAM_INVALID,
               "Std: CheckAxis failed: axis=%ld", static_cast<long>(axis));
    CHECK_COND(CheckOutputShape(self, axis, out), ACLNN_ERR_PARAM_INVALID,
               "Std: CheckOutputShape failed: axis=%ld", static_cast<long>(axis));
    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：参数检查 + 计算 workspace 大小
 *
 * 标准流程：
 * 1. CREATE_EXECUTOR()   - 创建执行器
 * 2. CheckParams()       - null / dtype / format / rank / axis / out-shape 检查（含显式拒绝分支）
 * 3. Contiguous()        - 非连续输入转换为连续 tensor（保证 kernel 访问 c_contiguous 数据）
 * 4. l0op::Std()         - 调用 L0 算子（InferShape -> IsAiCoreSupport -> AllocTensor -> Kernel）
 * 5. ViewCopy()          - 结果写回用户 out tensor
 * 6. GetWorkspaceSize()  - 返回 workspace 大小
 */
extern "C" aclnnStatus aclnnNumpyStdGetWorkspaceSize(
    const aclTensor* self,
    int64_t axis,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNumpyStd, DFX_IN(self), DFX_OUT(out));

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, axis, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const int64_t rank = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    int64_t normalizedAxis = 0;
    if (!NormalizeAxis(axis, rank, normalizedAxis)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Std: normalize axis failed after CheckParams, axis=%ld.",
                static_cast<long>(axis));
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::Std(selfContiguous, normalizedAxis, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief 第二段接口：执行计算
 *
 * 本接口只负责在给定 stream 上提交任务并返回状态码；显式同步（aclrtSynchronizeStream）
 * 与状态码检查由 adapter / 调用方完成，保证 kernel 真正在 NPU 上执行且无 host 回退。
 */
extern "C" aclnnStatus aclnnNumpyStd(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNumpyStd);

    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Std: aclnnNumpyStd received null executor.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (workspaceSize > 0U && workspace == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                "Std: aclnnNumpyStd received null workspace with workspaceSize=%lu.",
                static_cast<unsigned long>(workspaceSize));
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ret,
                "Std: aclnnNumpyStd execute failed, ret=%d, workspaceSize=%lu.",
                static_cast<int>(ret), static_cast<unsigned long>(workspaceSize));
    }
    return ret;
}