
#include <asnumpy/custom_ops/argpartition.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_argpartition.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

constexpr int64_t kArgpartitionMaxAxisLen = 32;

void ValidateNumpyArgpartitionProfile(const NPUArray& a, int64_t kth, int64_t axis, int64_t& normalizedAxis) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.argpartition received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument(
            "asnumpy.argpartition supports float32 input only; implicit casting is not allowed");
    }
    if (a.shape.size() != 2) {
        throw std::invalid_argument(
            "asnumpy.argpartition candidate admits rank 2 float32 input only");
    }
    if (a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument(
            "asnumpy.argpartition candidate rejects empty input");
    }
    if (a.shape[1] > kArgpartitionMaxAxisLen) {
        throw std::invalid_argument(
            "asnumpy.argpartition candidate admits axis length <= 32 only");
    }
    if (kth < 0 || kth >= a.shape[1]) {
        throw std::invalid_argument(
            "asnumpy.argpartition candidate requires 0 <= kth < a.shape[1]");
    }
    normalizedAxis = axis;
    if (normalizedAxis < 0) {
        normalizedAxis += static_cast<int64_t>(a.shape.size());
    }
    if (normalizedAxis != 1) {
        throw std::invalid_argument(
            "asnumpy.argpartition candidate admits only axis=1 or its rank-2 alias axis=-1");
    }
}
} // namespace

NPUArray NumpyArgpartition(const NPUArray& a, int64_t kth, int64_t axis) {
    int64_t normalizedAxis = 0;
    ValidateNumpyArgpartitionProfile(a, kth, axis, normalizedAxis);
    const std::vector<int64_t> outShape = a.shape;

    NPUArray result(outShape, ACL_INT64);

    // Resolve the vendor entry points from the vendor library itself; see
    // custom_opp_symbols.hpp for why a plain link is not enough.
    using GetWorkspaceSizeFn = decltype(&aclnnArgpartitionGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnArgpartition);
    const auto getWorkspaceSize =
        custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnArgpartitionGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnArgpartition");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspaceSize(
        a.tensorPtr, kth, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnArgpartitionGetWorkspaceSize");

    // The executor is marked repeatable before execution so that destroying it
    // after a successful run is safe on this CANN build.
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");

    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnArgpartition");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");

    return result;
}

} // namespace asnumpy
