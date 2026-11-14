#pragma once
#include <asnumpy/utils/npu_array.hpp>
#include <cstdint>
namespace asnumpy {
NPUArray NumpyRepeat(const NPUArray& a, int64_t repeats, int64_t axis = 1);
}
