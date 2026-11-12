/******************************************************************************
 * Copyright (c) 2026 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include <asnumpy/custom_ops/pad.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_pad.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

void ValidatePadProfile(const NPUArray& a, const std::vector<int64_t>& padWidth) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.pad received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument(
            "asnumpy.pad supports float32 input only; implicit casting is not allowed");
    }
    if (a.shape.size() != 2) {
        throw std::invalid_argument(
            "asnumpy.pad candidate admits rank 2 float32 input only");
    }
    if (a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.pad candidate rejects empty input");
    }
    if (padWidth.size() != 4) {
        throw std::invalid_argument(
            "asnumpy.pad candidate requires exactly four pad entries for rank 2 input");
    }
    for (int64_t value : padWidth) {
        if (value < 0) {
            throw std::invalid_argument(
                "asnumpy.pad candidate rejects negative pad widths");
        }
    }
    if (a.shape[0] + padWidth[0] + padWidth[1] <= 0 ||
        a.shape[1] + padWidth[2] + padWidth[3] <= 0) {
        throw std::invalid_argument(
            "asnumpy.pad candidate requires a positive output shape");
    }
}

// aclCreateIntArray allocates an ACL object; the executor copies the values
// during GetWorkspaceSize, so releasing it after execution is safe.
struct IntArrayGuard {
    aclIntArray* value;
    explicit IntArrayGuard(aclIntArray* array) : value(array) {}
    ~IntArrayGuard() {
        if (value != nullptr) {
            aclDestroyIntArray(value);
        }
    }
    IntArrayGuard(const IntArrayGuard&) = delete;
    IntArrayGuard& operator=(const IntArrayGuard&) = delete;
};

} // namespace

NPUArray Pad(const NPUArray& a, const std::vector<int64_t>& padWidth) {
    ValidatePadProfile(a, padWidth);
    const std::vector<int64_t> outShape = {
        a.shape[0] + padWidth[0] + padWidth[1],
        a.shape[1] + padWidth[2] + padWidth[3],
    };

    NPUArray result(outShape, a.dtype);

    IntArrayGuard pads(aclCreateIntArray(padWidth.data(), padWidth.size()));
    if (pads.value == nullptr) {
        throw std::runtime_error("asnumpy.pad could not create the pad-width array");
    }

    // Resolve the vendor entry points from the vendor library itself; see
    // custom_opp_symbols.hpp for why a plain link is not enough.
    using GetWorkspaceSizeFn = decltype(&aclnnPadGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnPad);
    const auto getWorkspaceSize =
        custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnPadGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnPad");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // The L0 entry admits only mode 0 (constant), matching mode='constant'.
    auto error = getWorkspaceSize(
        a.tensorPtr, pads.value, 0, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnPadGetWorkspaceSize");

    // The executor is marked repeatable before execution so that destroying it
    // after a successful run is safe on this CANN build.
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");

    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnPad");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");

    return result;
}

} // namespace asnumpy
