
#include <asnumpy/custom_ops/tril.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_tril.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

void ValidateNumpyTrilProfile(const NPUArray& a) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.tril received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument(
            "asnumpy.tril supports float32 input only; implicit casting is not allowed");
    }
    if (a.shape.size() != 2) {
        throw std::invalid_argument(
            "asnumpy.tril candidate admits rank 2 float32 input only");
    }
    if (a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument(
            "asnumpy.tril candidate rejects empty input");
    }
}
} // namespace

NPUArray NumpyTril(const NPUArray& a) {
    ValidateNumpyTrilProfile(a);
    const std::vector<int64_t> outShape = a.shape;

    NPUArray result(outShape, a.dtype);

    // Resolve the vendor entry points from the vendor library itself; see
    // custom_opp_symbols.hpp for why a plain link is not enough.
    using GetWorkspaceSizeFn = decltype(&aclnnTrilGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnTril);
    const auto getWorkspaceSize =
        custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnTrilGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnTril");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspaceSize(
        a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnTrilGetWorkspaceSize");

    // The executor is marked repeatable before execution so that destroying it
    // after a successful run is safe on this CANN build.
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");

    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnTril");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");

    return result;
}

} // namespace asnumpy
