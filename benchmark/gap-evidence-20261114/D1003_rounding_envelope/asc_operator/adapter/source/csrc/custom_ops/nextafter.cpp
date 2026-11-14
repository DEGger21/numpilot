#include <asnumpy/custom_ops/nextafter.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_nextafter.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

void ValidateNextafterProfile(const NPUArray& a, const NPUArray& b) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.nextafter received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument("asnumpy.nextafter supports float32 input only; implicit casting is not allowed");
    }
    if (b.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.nextafter received a null target tensor");
    }
    if (b.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument("asnumpy.nextafter supports float32 target only");
    }
    if (a.shape != b.shape) {
        throw std::invalid_argument("asnumpy.nextafter requires identical input shapes");
    }
    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.nextafter admits non-empty rank 2 input only");
    }
}

} // namespace

NPUArray NumpyNextafter(const NPUArray& a, const NPUArray& b) {
    const std::vector<int64_t> outShape = a.shape;
    ValidateNextafterProfile(a, b);

    NPUArray result(outShape, a.dtype);
    using GetWorkspaceSizeFn = decltype(&aclnnNextafterGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnNextafter);
    const auto getWorkspaceSize = custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnNextafterGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnNextafter");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspaceSize(a.tensorPtr, b.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnNextafterGetWorkspaceSize");
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");
    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnNextafter");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");
    return result;
}

} // namespace asnumpy
