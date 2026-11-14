#include <asnumpy/custom_ops/diff.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_diff.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

void ValidateDiffProfile(const NPUArray& a, int64_t normalizedAxis) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.diff received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument("asnumpy.diff supports float32 input only; implicit casting is not allowed");
    }
    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] < 2) {
        throw std::invalid_argument("asnumpy.diff admits rank 2 input with at least two columns");
    }
    if (normalizedAxis != 1) {
        throw std::invalid_argument("asnumpy.diff admits only axis=1 or axis=-1");
    }
}

} // namespace

NPUArray NumpyDiff(const NPUArray& a, int64_t axis) {
    int64_t normalizedAxis = axis < 0 ? axis + static_cast<int64_t>(a.shape.size()) : axis;
    std::vector<int64_t> outShape = a.shape;
    outShape[1] -= 1;
    ValidateDiffProfile(a, normalizedAxis);

    NPUArray result(outShape, a.dtype);
    using GetWorkspaceSizeFn = decltype(&aclnnDiffGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnDiff);
    const auto getWorkspaceSize = custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnDiffGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnDiff");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspaceSize(a.tensorPtr, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnDiffGetWorkspaceSize");
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");
    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnDiff");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");
    return result;
}

} // namespace asnumpy
