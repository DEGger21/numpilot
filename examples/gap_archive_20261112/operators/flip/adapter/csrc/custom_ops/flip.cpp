
#include <asnumpy/custom_ops/flip.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_flip.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

constexpr int64_t kAdmittedAxis = 1;

void ValidateNumpyFlipProfile(const NPUArray& a, int64_t axis, int64_t& normalizedAxis) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.flip received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument(
            "asnumpy.flip supports float32 input only; implicit casting is not allowed");
    }
    if (a.shape.size() != 2) {
        throw std::invalid_argument(
            "asnumpy.flip candidate admits rank 2 float32 input only");
    }
    if (a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument(
            "asnumpy.flip candidate rejects empty input");
    }
    normalizedAxis = axis;
    if (normalizedAxis < 0) {
        normalizedAxis += static_cast<int64_t>(a.shape.size());
    }
    if (normalizedAxis != kAdmittedAxis) {
        throw std::invalid_argument(
            "asnumpy.flip candidate admits only axis=1 or its rank-2 alias axis=-1");
    }
}
} // namespace

NPUArray NumpyFlip(const NPUArray& a, int64_t axis) {
    int64_t normalizedAxis = 0;
    ValidateNumpyFlipProfile(a, axis, normalizedAxis);
    const std::vector<int64_t> outShape = a.shape;

    NPUArray result(outShape, a.dtype);

    // Resolve the vendor entry points from the vendor library itself; see
    // custom_opp_symbols.hpp for why a plain link is not enough.
    using GetWorkspaceSizeFn = decltype(&aclnnFlipGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnFlip);
    const auto getWorkspaceSize =
        custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnFlipGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnFlip");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspaceSize(
        a.tensorPtr, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnFlipGetWorkspaceSize");

    // The executor is marked repeatable before execution so that destroying it
    // after a successful run is safe on this CANN build.
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");

    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnFlip");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");

    return result;
}

} // namespace asnumpy
