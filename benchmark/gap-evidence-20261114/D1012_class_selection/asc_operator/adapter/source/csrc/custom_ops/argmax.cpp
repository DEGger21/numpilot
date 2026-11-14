#include <asnumpy/custom_ops/argmax.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_argmax.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

void ValidateArgmaxProfile(const NPUArray& a, int64_t normalizedAxis) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.argmax received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument("asnumpy.argmax supports float32 input only; implicit casting is not allowed");
    }
    if (normalizedAxis != 1) {
        throw std::invalid_argument("asnumpy.argmax admits only axis=1 or axis=-1");
    }
    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.argmax admits non-empty rank 2 input only");
    }
}

} // namespace

NPUArray NumpyArgmax(const NPUArray& a, int64_t axis) {
    int64_t normalizedAxis = axis < 0 ? axis + static_cast<int64_t>(a.shape.size()) : axis;
    const std::vector<int64_t> outShape{a.shape[0]};
    ValidateArgmaxProfile(a, normalizedAxis);

    NPUArray result(outShape, ACL_INT64);
    using GetWorkspaceSizeFn = decltype(&aclnnArgmaxGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnArgmax);
    const auto getWorkspaceSize = custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnArgmaxGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnArgmax");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspaceSize(a.tensorPtr, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnArgmaxGetWorkspaceSize");
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");
    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnArgmax");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");
    return result;
}

} // namespace asnumpy
