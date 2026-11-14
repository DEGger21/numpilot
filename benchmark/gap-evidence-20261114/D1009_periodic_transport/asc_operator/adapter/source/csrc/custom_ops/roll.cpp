#include <asnumpy/custom_ops/roll.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_roll.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

void ValidateRollProfile(const NPUArray& a, int64_t shift, int64_t normalizedAxis) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.roll received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument("asnumpy.roll supports float32 input only; implicit casting is not allowed");
    }
    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.roll admits non-empty rank 2 input only");
    }
    if (shift != 1) {
        throw std::invalid_argument("asnumpy.roll candidate admits only shift=1");
    }
    if (normalizedAxis != 1) {
        throw std::invalid_argument("asnumpy.roll admits only axis=1 or axis=-1");
    }
}

} // namespace

NPUArray NumpyRoll(const NPUArray& a, int64_t shift, int64_t axis) {
    int64_t normalizedAxis = axis < 0 ? axis + static_cast<int64_t>(a.shape.size()) : axis;
    const std::vector<int64_t> outShape = a.shape;
    ValidateRollProfile(a, shift, normalizedAxis);

    NPUArray result(outShape, a.dtype);
    using GetWorkspaceSizeFn = decltype(&aclnnRollGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnRoll);
    const auto getWorkspaceSize = custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnRollGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnRoll");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspaceSize(a.tensorPtr, shift, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnRollGetWorkspaceSize");
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");
    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnRoll");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");
    return result;
}

} // namespace asnumpy
