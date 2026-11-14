#pragma once
#include <asnumpy/utils/npu_array.hpp>
#include <cstdint>
namespace asnumpy {
NPUArray NumpyArgmax(const NPUArray& a, int64_t axis = 1);
}
