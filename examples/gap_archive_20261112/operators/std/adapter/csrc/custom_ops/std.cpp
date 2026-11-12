
#include <asnumpy/custom_ops/std.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_numpy_std.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

constexpr int64_t kAdmittedAxis = 1;

void ValidateNumpyStdProfile(const NPUArray& a, int64_t axis, int64_t& normalizedAxis) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.std received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument(
            "asnumpy.std supports float32 input only; implicit casting is not allowed");
    }
    if (a.shape.size() != 2) {
        throw std::invalid_argument(
            "asnumpy.std candidate admits rank 2 float32 input only");
    }
    if (a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument(
            "asnumpy.std candidate rejects empty input");
    }
    normalizedAxis = axis;
    if (normalizedAxis < 0) {
        normalizedAxis += static_cast<int64_t>(a.shape.size());
    }
    if (normalizedAxis != kAdmittedAxis) {
        throw std::invalid_argument(
            "asnumpy.std candidate admits only axis=1 or its rank-2 alias axis=-1");
    }
}
} // namespace

NPUArray NumpyStd(const NPUArray& a, int64_t axis) {
    int64_t normalizedAxis = 0;
    ValidateNumpyStdProfile(a, axis, normalizedAxis);
    const std::vector<int64_t> outShape = {a.shape[0]};

    NPUArray result(outShape, a.dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnNumpyStdGetWorkspaceSize(
        a.tensorPtr, normalizedAxis, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnNumpyStdGetWorkspaceSize");

    // The executor is marked repeatable before execution so that destroying it
    // after a successful run is safe on this CANN build.
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");

    AclWorkspace workspace(workspaceSize);
    error = aclnnNumpyStd(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnNumpyStd");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");

    return result;
}

} // namespace asnumpy
