#include <asnumpy/custom_ops/spacing.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_spacing.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

void ValidateSpacingProfile(const NPUArray& a) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.spacing received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument("asnumpy.spacing supports float32 input only; implicit casting is not allowed");
    }
    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.spacing admits non-empty rank 2 input only");
    }
}

} // namespace

NPUArray NumpySpacing(const NPUArray& a) {
    const std::vector<int64_t> outShape = a.shape;
    ValidateSpacingProfile(a);

    NPUArray result(outShape, a.dtype);
    using GetWorkspaceSizeFn = decltype(&aclnnSpacingGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnSpacing);
    const auto getWorkspaceSize = custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnSpacingGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnSpacing");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspaceSize(a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnSpacingGetWorkspaceSize");
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");
    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnSpacing");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");
    return result;
}

} // namespace asnumpy
