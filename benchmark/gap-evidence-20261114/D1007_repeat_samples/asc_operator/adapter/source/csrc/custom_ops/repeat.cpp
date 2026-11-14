#include <asnumpy/custom_ops/repeat.hpp>
#include <asnumpy/custom_ops/custom_opp_symbols.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_repeat.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

void ValidateRepeatProfile(const NPUArray& a, int64_t repeats, int64_t normalizedAxis) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.repeat received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument("asnumpy.repeat supports float32 input only; implicit casting is not allowed");
    }
    if (normalizedAxis != 1) {
        throw std::invalid_argument("asnumpy.repeat admits only axis=1 or axis=-1");
    }
    if (repeats != 3) {
        throw std::invalid_argument("asnumpy.repeat candidate admits only repeats=3");
    }
    if (a.shape.size() != 2 || a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument("asnumpy.repeat admits non-empty rank 2 input only");
    }
}

} // namespace

NPUArray NumpyRepeat(const NPUArray& a, int64_t repeats, int64_t axis) {
    int64_t normalizedAxis = axis < 0 ? axis + static_cast<int64_t>(a.shape.size()) : axis;
    std::vector<int64_t> outShape = a.shape;
    outShape[1] *= repeats;
    ValidateRepeatProfile(a, repeats, normalizedAxis);

    NPUArray result(outShape, a.dtype);
    using GetWorkspaceSizeFn = decltype(&aclnnRepeatGetWorkspaceSize);
    using ExecuteFn = decltype(&aclnnRepeat);
    const auto getWorkspaceSize = custom_opp::LoadSymbol<GetWorkspaceSizeFn>("aclnnRepeatGetWorkspaceSize");
    const auto execute = custom_opp::LoadSymbol<ExecuteFn>("aclnnRepeat");

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspaceSize(a.tensorPtr, repeats, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnRepeatGetWorkspaceSize");
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");
    AclWorkspace workspace(workspaceSize);
    error = execute(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnRepeat");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");
    return result;
}

} // namespace asnumpy
