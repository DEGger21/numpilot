
#include <asnumpy/custom_ops/ediff1d.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnn_ediff1d.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace asnumpy {
namespace {

void ValidateNumpyEdiff1dProfile(const NPUArray& a, int64_t& numel) {
    if (a.tensorPtr == nullptr) {
        throw std::invalid_argument("asnumpy.ediff1d received a null input tensor");
    }
    if (a.aclDtype != ACL_FLOAT) {
        throw std::invalid_argument(
            "asnumpy.ediff1d supports float32 input only; implicit casting is not allowed");
    }
    if (a.shape.size() != 2) {
        throw std::invalid_argument(
            "asnumpy.ediff1d candidate admits rank 2 float32 input only");
    }
    if (a.shape[0] <= 0 || a.shape[1] <= 0) {
        throw std::invalid_argument(
            "asnumpy.ediff1d candidate rejects empty input");
    }
    numel = a.shape[0] * a.shape[1];
    if (numel < 2) {
        throw std::invalid_argument(
            "asnumpy.ediff1d candidate requires at least two input elements");
    }
}
} // namespace

NPUArray NumpyEdiff1d(const NPUArray& a) {
    int64_t numel = 0;
    ValidateNumpyEdiff1dProfile(a, numel);
    const std::vector<int64_t> outShape = {numel - 1};

    NPUArray result(outShape, a.dtype);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnEdiff1dGetWorkspaceSize(
        a.tensorPtr, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnEdiff1dGetWorkspaceSize");

    // The executor is marked repeatable before execution so that destroying it
    // after a successful run is safe on this CANN build.
    error = aclSetAclOpExecutorRepeatable(executor);
    ACLNN_CHECK(error, "aclSetAclOpExecutorRepeatable");

    AclWorkspace workspace(workspaceSize);
    error = aclnnEdiff1d(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnEdiff1d");

    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");

    error = aclDestroyAclOpExecutor(executor);
    ACLNN_CHECK(error, "aclDestroyAclOpExecutor");

    return result;
}

} // namespace asnumpy
