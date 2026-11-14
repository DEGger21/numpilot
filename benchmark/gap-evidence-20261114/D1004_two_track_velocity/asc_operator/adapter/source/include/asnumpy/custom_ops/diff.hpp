#pragma once
#include <asnumpy/utils/npu_array.hpp>
#include <cstdint>
namespace asnumpy {
NPUArray NumpyDiff(const NPUArray& a, int64_t axis = 1);
}
